package org.hjackson.llm;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.stream.IntStream;

public class GPT2 {

    private static final Logger log = LoggerFactory.getLogger(GPT2.class);
    final float EPSILON = 1e-5F;
    private final AtomicBoolean activationsMem = new AtomicBoolean(false);
    private ExecutorService executorService = Executors.newFixedThreadPool(1000);
    private final List<Callable<Object>> ignoreAll = new ArrayList<>(128);

    private final AtomicLong gpt2_forward_counter = new AtomicLong();
    public final AtomicLong gpt2_forward_counter_layer = new AtomicLong();
    private final AtomicLong gpt2_backward_counter_layer = new AtomicLong();

    private final AtomicLong matmul_forward_counter = new AtomicLong();

    public static final int GPT2_EOT = 50256;

    private static float GELU_SCALING_FACTOR = (float) Math.sqrt(2.0f / Math.PI);
    public final GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    public final ParameterTensors params;
    private int num_parameters;
    // gradients of the weights
    public ParameterTensors grads;
    private float grads_memory;
    // buffers for the AdamW optimizer
    private float[] m_memory;
    private float[] v_memory;
    // the activations of the model, and their sizes
    public ActivationTensors acts;
    private int num_activations;
    // gradients of the activations
    private ActivationTensors grads_acts;
    private float grads_acts_memory;
    // other run state configuration
    private int batch_size; // the batch size (B) of current forward pass
    private int seq_len; // the sequence length (T) of current forward pass
    private int[] cacheInputs; // the input tokens for the current forward pass
    private int[] cacheTargets; // the target tokens for the current forward pass
    private DataLoader loader;
    //private DataLoader train_loader;
    public float mean_loss = 0.0f; // after a forward pass with targets, will be populated with the mean loss
    private final long file_size;
    private final Arena memoryArena;
    private final MemorySegment data;
    private static final int headerSize = 256 * 4;// bytes
    private int model_header[] = new int[headerSize];
    private IntBuffer header = IntBuffer.allocate(256);
    private MemorySegment mappedFile;

    public GPT2(String checkpoint_path) throws Exception {
        try (FileChannel fileChannel = FileChannel.open(Paths.get(checkpoint_path),
                StandardOpenOption.READ)) {
            this.file_size = fileChannel.size();
            this.memoryArena = Arena.ofAuto();
            log.info("File Size: {}", file_size);
            mappedFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, this.file_size, this.memoryArena);
            this.data = mappedFile;
            alloc_header(mappedFile, header);
            if (header.get(0) != 20240326) {
                throw new Exception("Bad magic model file");
            }
            if (header.get(1) != 1) {
                throw new Exception("Bad version in model file");
            }
            config = new GPT2Config(header);
            assert (config.vocab_size > 0);
            params = new ParameterTensors(mappedFile, config);
            gpt2_build_from_checkpoint(checkpoint_path);
        }
        final float wte0 = params.getMem(0);
        Assert.floatEquals(wte0, -0.11010301f);
    }

    public void gpt2_build_from_checkpoint(final String checkpoint_path) throws Exception {

        int maxT, V, L, NH, C;
        this.config.max_seq_len = maxT = header.get(2);
        this.config.vocab_size = V = header.get(3);
        this.config.num_layers = L = header.get(4);
        this.config.num_heads = NH = header.get(5);
        this.config.channels = C = header.get(6);

        log.info("[GPT-2]\n");
        log.info("max_seq_len: " + maxT);
        log.info("vocab_size: " + V);
        log.info("num_layers: " + L);
        log.info("num_heads: " + NH);
        log.info("channels: " + C);

        log.info("num_parameters: {}", params.getNumParams());
        this.num_parameters = params.getNumParams();

        // read in all the parameters from file
        // this.params_memory = malloc_and_point_parameters(this.params, 256, is);

        // other inits
        //this.acts_memory = 0;
        this.grads_memory = 0;
        this.m_memory = null;
        this.v_memory = null;
        this.grads_acts_memory = 0;
        this.batch_size = 0;
        this.seq_len = 0;
        this.mean_loss = -1.0f; // -1.0f will designate no loss
    }

    public void alloc_header(MemorySegment mappedFile, IntBuffer header) throws Exception {
        int startPos = 0;
        int endPos = headerSize;

        IntBuffer tmp = mappedFile.asSlice(startPos, endPos).asByteBuffer()
                .order(ByteOrder.LITTLE_ENDIAN)
                .asIntBuffer();
        //tmp is a view into the mapped file so we need to copy it
        log.info("intBuffer size: {}", tmp.capacity());
        header.put(tmp);
        log.info("header[0]={}", header.get(0));
        log.info("header[1]={}", header.get(1));
    }

    public FloatBuffer takefloats(MemorySegment memorySegment, long[] position, int... dims) {
        long totalBytes = 1;
        for (int d : dims) {
            totalBytes *= d;
        }
        totalBytes *= Float.BYTES;
        MemorySegment slice = memorySegment.asSlice(position[0], totalBytes);
        position[0] += totalBytes;
        return slice.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    }

    void gpt2_zero_grad() {
        if (grads_acts != null) {
            grads_acts.zeroFill();
        }
        if (grads != null) {
            grads.zeroFill();
        }
    }


    private void encoder_backward(int dwte, int dwpe, int dout, DataLoader inp, int B, int T, int C) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int dout_bt = dout + b * T * C + t * C;//grads_acts
                int ix = loader.getInputs(b * T + t);
                int dwte_ix = dwte + ix * C;//grads
                int dwpe_t = dwpe + t * C;//grads
                for (int i = 0; i < C; i++) {
                    //float d = dout_bt[i];
                    float d = grads_acts.getMem(dout_bt + i);

                    //dwte_ix[i] = dwte_ix[i] + d;
                    float dwte_ix_i = grads.getMem(dwte_ix + i);
                    grads.setMem(dwte_ix + i, dwte_ix_i + d);

                    //dwpe_t[i] = dwpe_t[i] + d;
                    float dwpe_t_i = grads.getMem(dwpe_t + i);
                    grads.setMem(dwpe_t + i, dwpe_t_i + d);
                }
            }
        }
    }

    void gpt2_update(float learning_rate, float beta1, float beta2, float eps, float weight_decay,
                     int t) {
        // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

        // lazily allocate the memory for m_memory and v_memory
        if (m_memory == null) {
            //m_memory = (float*)calloc(model->num_parameters, sizeof(float));
            //v_memory = (float*)calloc(model->num_parameters, sizeof(float));
            m_memory = new float[(int) num_parameters];
            v_memory = new float[(int) num_parameters];
        }

        for (int i = 0; i < num_parameters; i++) {
            float param = params.getMem(i);
            float grad = grads.getMem(i);

            // update the first moment (momentum)
            float m = beta1 * m_memory[i] + (1.0f - beta1) * grad;
            // update the second moment (RMSprop)
            float v = beta2 * v_memory[i] + (1.0f - beta2) * grad * grad;
            // bias-correct both moments
            float m_hat = (float) (m / (1.0f - Math.pow(beta1, t)));
            float v_hat = (float) (v / (1.0f - Math.pow(beta2, t)));

            // update
            m_memory[i] = m;
            v_memory[i] = v;

            //params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
            float tmp = (float) (learning_rate * (m_hat / (Math.sqrt(v_hat) + eps)
                    + weight_decay * param));
            float pm_i = params.getMem(i);
            params.setMem(i, pm_i - tmp);
        }
    }

    // grads_acts, grads_acts, grads_acts, grads_acts, acts, acts
    void attention_backward(int dinp, int dpreatt, int datt, int dout, int inp, int att, int B, int T,
                            int C, int NH) {
        // inp/dinp are (B, T, 3C) Q,K,V
        // att/datt/dpreatt are (B, NH, T, T)
        // dout is (B, T, C)
        int C3 = C * 3;
        int hs = C / NH; // head size
        float scale = (float) (1.0 / Math.sqrt(hs));

        float tmp;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int h = 0; h < NH; h++) {
                    int att_bth = att + b * NH * T * T + h * T * T + t * T; //acts
                    int datt_bth = datt + b * NH * T * T + h * T * T + t * T; //grads_acts
                    int dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;// grads_acts
                    int dquery_t = dinp + b * T * C3 + t * C3 + h * hs;//grads_acts
                    int query_t = inp + b * T * C3 + t * C3 + h * hs;//acts

                    // backward pass 4, through the value accumulation
                    int dout_bth = dout + b * T * C + t * C + h * hs;//grads_acts
                    for (int t2 = 0; t2 <= t; t2++) {
                        int value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value // acts.
                        int dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2;//grads_acts
                        for (int i = 0; i < hs; i++) {
                            // in the forward pass this was:
                            // out_bth[i] += att_bth[t2] * value_t2[i];
                            // so now we have:
                            float dout_bth_i = grads_acts.getMem(dout_bth + i);
                            float va_x_do = acts.getMem(value_t2 + i) * dout_bth_i;
                            grads_acts.setMem(datt_bth + t2, grads_acts.getMem(datt_bth + t2) + acts.getMem(value_t2 + i) * dout_bth_i);
                            grads_acts.setMem(dvalue_t2 + i, grads_acts.getMem(dvalue_t2 + i) + acts.getMem(att_bth + t2) * dout_bth_i);
                        }
                    }

                    // backward pass 2 & 3, the softmax
                    // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                    for (int t2 = 0; t2 <= t; t2++) {
                        for (int t3 = 0; t3 <= t; t3++) {
                            float indicator = t2 == t3 ? 1.0f : 0.0f;
                            //float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                            //dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                            float att_bh_t2 = acts.getMem(att_bth + t2);
                            float att_bh_t3 = acts.getMem(att_bth + t3);
                            float datt_bth_t2 = grads_acts.getMem(datt_bth + t2);
                            float local_derivative = att_bh_t2 * (indicator - att_bh_t3);
                            grads_acts.setMem(dpreatt_bth + t3, grads_acts.getMem(dpreatt_bth + t3) + local_derivative * datt_bth_t2);
                        }
                    }

                    // backward pass 1, the query @ key matmul
                    for (int t2 = 0; t2 <= t; t2++) {
                        int key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key   // acts
                        int dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key // grads_acts
                        for (int i = 0; i < hs; i++) {
                            // in the forward pass this was:
                            float key_t2_i = acts.getMem(key_t2 + i);
                            float dpreatt_bth_t2 = grads_acts.getMem(dpreatt_bth + t2);
                            float dquery_t_i = grads_acts.getMem(dquery_t + i);

                            tmp = dquery_t_i + key_t2_i * dpreatt_bth_t2 * scale;
                            grads_acts.setMem(dquery_t + i, tmp);

                            float dkey_t2_i = grads_acts.getMem(dkey_t2 + i);
                            float query_t_i = acts.getMem(query_t + i);
                            tmp = dkey_t2_i + query_t_i * dpreatt_bth_t2 * scale;
                            grads_acts.setMem(dkey_t2 + i, tmp);
                        }
                    }
                }
            }
        }


    }

    // grads_acts, acts,  grads_acts
    private void gelu_backward(int dinp, int inp, int dout, int N) {
        for (int i = 0; i < N; i++) {
            //float x = inp[i];
            float x = acts.getMem(inp + i);
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = (float) Math.tanh(tanh_arg);
            float coshf_out = (float) Math.cosh(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad =
                    0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out
                            * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);

            //dinp[i] = dinp[i] + local_grad * dout[i];
            float tmp = local_grad * grads_acts.getMem(dout + i);
            float dinp_i = grads_acts.getMem(dinp + i) + tmp;
            grads_acts.setMem(dinp + i, dinp_i);
        }
    }

  /*
  dinp1 == grads_acts
  dinp2 == grads_acts
  dout  == grads_acts
   */

    private void residual_backward(int dinp1, int dinp2, int dout, int N) {
        for (int i = 0; i < N; i++) {

            //dinp1[i] = dinp1[i] + dout[i];
            float tmp = grads_acts.getMem(dinp1 + i)
                    + grads_acts.getMem(dout + i);
            grads_acts.setMem(dinp1 + i, tmp);

            //dinp2[i] = dinp2[i] + dout[i];
            tmp = grads_acts.getMem(dinp2 + i)
                    + grads_acts.getMem(dout + i);
            grads_acts.setMem(dinp2 + i, tmp);
        }
    }

    // dinp == grads_acts
    // dweight == grads
    // dbias == grads
    // dout == grads_acts
    // inp == acts
    // weight == params
    // mean == acts
    // rstd == acts              //grads_acts, grads,      grads    , grads_acts, acts , params    , acts    , acts
    private void layernorm_backward(int dinp, int dweight, int dbias, int dout, int inp, int weight, int mean, int rstd, int B, int T, int C) {

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int dout_bt = dout + b * T * C + t * C;//grads_acts
                int inp_bt = inp + b * T * C + t * C;//acts
                int dinp_bt = dinp + b * T * C + t * C;//grads_acts
                //float mean_bt = mean[b * T + t];//acts
                //float rstd_bt = rstd[b * T + t];//acts

                float mean_bt = acts.getMem(mean + b * T + t);//acts
                float rstd_bt = acts.getMem(rstd + b * T + t);//acts

                // first: two reduce operations
                float dnorm_mean = 0.0f;
                float dnorm_norm_mean = 0.0f;
                for (int i = 0; i < C; i++) {
                    //float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                    float norm = acts.getMem(inp_bt + i) - mean_bt;
                    float norm_bti = norm * rstd_bt;

                    //float dnorm_i = weight[i] * dout_bt[i];
                    float weight_i = params.getMem(weight + i);
                    float dout_bt_i = grads_acts.getMem(dout_bt + i);
                    float dnorm_i = weight_i * dout_bt_i;
                    dnorm_mean += dnorm_i;
                    dnorm_norm_mean += dnorm_i * norm_bti;
                }
                dnorm_mean = dnorm_mean / C;
                dnorm_norm_mean = dnorm_norm_mean / C;

                // now iterate again and accumulate all the gradients
                for (int i = 0; i < C; i++) {
                    float norm_bti = (acts.getMem(inp_bt + i) - mean_bt) * rstd_bt;
                    float dout_bt_i = grads_acts.getMem(dout_bt + i);
                    float dnorm_i = params.getMem(weight + i) * dout_bt_i;
                    // gradient contribution to bias
                    grads.setMem(dbias + i, grads.getMem(dbias + i) + dout_bt_i);
                    grads.setMem(dweight + i, grads.getMem(weight + i) + norm_bti * dout_bt_i);
                    // gradient contribution to input
                    float dval = 0.0f;
                    dval += dnorm_i; // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt; // final scale
                    grads_acts.setMem(dinp_bt + i, grads_acts.getMem(dinp_bt + i) + dval);
                }
            }
        }

    }

    // grads_acts, grads,    grads    , grads_acts, acts,   params
    // grads_acts, grads,    MIN_VALUE, grads_acts, acts,   params
    private void matmul_backward(int dinp, int dweight, int dbias, int dout, int inp, int weight,
                                 int B, int T, int C, int OC) {
        // most of the running time is spent here and in matmul_forward
        // this backward could be done in a single "round" of loops
        // but that doesn't afford an efficient parallelization strategy

        // backward into inp first, parallelize over B,T
        //#pragma omp parallel for collapse(2)
        final int btMax = B * T;
        IntStream.range(0, btMax) // This is probably not the fastest way to thread this loop
                .parallel()
                .forEach(bt -> {
                    int b = bt / T;
                    int t = bt % T;

                    final int dout_bt = dout + b * T * OC + t * OC;//grads_acts
                    final int dinp_bt = dinp + b * T * C + t * C;//grads_acts
                    for (int o = 0; o < OC; o++) {
                        int wrow = weight + o * C;//params
                        float d = grads_acts.getMem(dout_bt + o);
                        for (int i = 0; i < C; i++) {
                            float row = params.getMem(wrow + i) * d;
                            float tmp = grads_acts.getMem(dinp_bt + i) + row;
                            grads_acts.setMem(dinp_bt + i, tmp);
                        }
                    }
                });
        // backward into weight/bias, parallelize over output channels OC
        //#pragma omp parallel for
        IntStream.range(0, OC) // This is probably not the fastest way to thread this loop
                .parallel()
                .forEach(o -> {
                    for (int b = 0; b < B; b++) {

                        for (int t = 0; t < T; t++) {
                            int dout_bt = dout + b * T * OC + t * OC;//grads_acts
                            int inp_bt = inp + b * T * C + t * C;//acts
                            int dwrow = dweight + o * C;//grads
                            float d = grads_acts.getMem(dout_bt + o);

                            if (dbias != Integer.MIN_VALUE) {
                                float tmp = grads.getMem(dbias + o) + d;//grads
                                grads.setMem(dbias + o, tmp);
                            }
                            for (int i = 0; i < C; i++) {
                                float row = acts.getMem(inp_bt + i) * d;
                                float tmp = grads.getMem(dwrow + i) + row;
                                grads.setMem(dwrow + i, tmp);
                            }
                        }
                    }
                });
    }


    private void matmul_backward_slow(int dinp, int dweight, int dbias, int dout, int inp, int weight,
                                      int B, int T, int C, int OC) {
        // most of the running time is spent here and in matmul_forward
        // this backward could be done in a single "round" of loops
        // but that doesn't afford an efficient parallelization strategy
        // backward into inp first, parallelize over B,T

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                final int dout_bt = dout + b * T * OC + t * OC;//grads_acts
                final int dinp_bt = dinp + b * T * C + t * C;//grads_acts
                for (int o = 0; o < OC; o++) {
                    int wrow = weight + o * C;//params
                    float d = grads_acts.getMem(dout_bt + o);

                    for (int i = 0; i < C; i++) {
                        float row = params.getMem(wrow + i) * d;
                        float tmp = grads_acts.getMem(dinp_bt + i) + row;
                        grads_acts.setMem(dinp_bt + i, tmp);
                    }
                }
            }
        }
        // backward into weight/bias, parallelize over output channels OC
        for (int o = 0; o < OC; o++) {
            for (int b = 0; b < B; b++) {
                for (int t = 0; t < T; t++) {
                    int dout_bt = dout + b * T * OC + t * OC;//grads_acts
                    int inp_bt = inp + b * T * C + t * C;//acts
                    int dwrow = dweight + o * C;//grads
                    float d = grads_acts.getMem(dout_bt + o);
                    if (dbias != Integer.MIN_VALUE) {
                        float tmp = grads.getMem(dbias + o) + d;//grads
                        grads.setMem(dbias + o, tmp);
                    }
                    for (int i = 0; i < C; i++) {
                        float row = acts.getMem(inp_bt + i) * d;
                        float tmp = grads.getMem(dwrow + i) + row;
                        grads.setMem(dwrow + i, tmp);
                    }
                }
            }
        }
    }

    private void crossentropy_softmax_backward(int dlogits, int dlosses, int probs,
                                               DataLoader targets, int B, int T, int V) {
        // backwards through both softmax and crossentropy
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int dlogits_bt = dlogits + b * T * V + t * V;
                int probs_bt = probs + b * T * V + t * V;
                float dloss = grads_acts.getMem(dlosses + b * T + t);
                int ix = loader.getTargets(b * T + t);
                for (int i = 0; i < V; i++) {
                    float p = acts.getMem(probs_bt + i);
                    float indicator = i == ix ? 1.0f : 0.0f;
                    float tmp = grads_acts.getMem(dlogits_bt + i) + (p - indicator) * dloss;
                    grads_acts.setMem(dlogits_bt + i, tmp);
                }
            }
        }
    }

    private static long random_u32(long state) {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return (state * 0x2545F4914F6CDD1DL) >> 32;
    }

    public static float random_f32(long state) { // random float32 in [0,1)
        return (random_u32(state) >> 8) / 16777216.0f;
    }

    public int sample_mult(int probabilities, int n, float coin) {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += acts.getMem(probabilities + i);
            if (coin < cdf) {
                return i;
            }
        }
        return n - 1; // in case of rounding errors
    }

    public void gpt2_forward(DataLoader loader, final int B, final int T) {
        gpt2_forward_counter.incrementAndGet();
        this.loader = loader;
        // ensure the model was initialized or error out
        if (!params.ok()) {
            throw new IllegalStateException("Error: model was not initialized properly.\n");
        }
        // convenience parameters
        int V = config.vocab_size;
        int L = config.num_layers;
        int NH = config.num_heads;
        int C = config.channels;
        // validate inputs, all indices must be in the range [0, V)
        for (int i = 0; i < B * T; i++) {
            assert (0 <= loader.getInputs(i) && loader.getInputs(i) < V);
            if (loader.targetsPresent()) {
                assert (0 <= loader.getTargets(i) && loader.getTargets(i) < V);
            }
        }
        // allocate space for all the activations if needed (done here, lazily)
        if (!activationsMem.get()) {
            activationsMem.set(true);
            batch_size = B;
            seq_len = T;
            acts = new ActivationTensors(config, B, T);
            if (gpt2_forward_counter.get() == 1L) {
                Assert.floatEquals(acts.getMem(acts.getResidual3()), 0.0f);
            }
            num_activations = acts.getNumActivations();
            log.info("num_activations: {}\n", num_activations);
            // also create memory for caching inputs and targets
        } else {
            // validate B,T is consistent with how we've allocated the memory before
            // in principle we could get more clever here in the future, for now this is safest
            if (B != batch_size || T != seq_len) {
                log.error("Model: B={} T={}, Desired: B={} T={}\n", batch_size, seq_len, B, T);
                System.exit(1);
            }
        }
        if (gpt2_forward_counter.get() == 1L) {
            Assert.floatEquals(acts.getMem(acts.getResidual3()), 0.0f);
        }
        int residual;
        // encoding goes into residual[0]
        encoder_forward(acts.getEncoded(), loader, params.getWte(), params.getWpe(), B, T, C);// checked 1
        for (int l = 0; l < L; l++) {
            long layerCount = gpt2_forward_counter_layer.incrementAndGet();
            //residual = l == 0 ? acts.encoded : acts.residual3[] + (l-1) * B * T * C;
            if (l == 0) {
                residual = acts.getEncoded();
            } else {
                residual = acts.getResidual3() + (l - 1) * B * T * C;
            }
            //System.out.printf("f==%d b==%d residual == %1.7f\n", layerCount, gpt2_backward_counter_layer.get(), acts.getMem(residual));
            if (gpt2_forward_counter.get() == 1L && l == 0) {
                Assert.floatEquals(acts.getMem(acts.getResidual3()), 0.0f);
            }
            // get the pointers of the weights for this layer
            int l_ln1w = params.getLn1w() + l * C;
            int l_ln1b = params.getLn1b() + l * C;
            int l_qkvw = params.getQkvw() + l * 3 * C * C;
            int l_qkvb = params.getQkvb() + l * 3 * C;
            int l_attprojw = params.getAttprojw() + l * C * C;
            int l_attprojb = params.getAttprojb() + l * C;
            int l_ln2w = params.getLn2w() + l * C;
            int l_ln2b = params.getLn2b() + l * C;
            int l_fcw = params.getFcw() + l * 4 * C * C;
            int l_fcb = params.getFcb() + l * 4 * C;
            int l_fcprojw = params.getFcprojw() + l * C * 4 * C;
            int l_fcprojb = params.getFcprojb() + l * C;
            // get the pointers of the activations for this layer
            int l_ln1 = acts.getLn1() + l * B * T * C;
            int l_ln1_mean = acts.getLn1Mean() + l * B * T;
            int l_ln1_rstd = acts.getLn1Rstd() + l * B * T;
            int l_qkv = acts.getQkv() + l * B * T * 3 * C;
            int l_atty = acts.getAtty() + l * B * T * C;
            int l_preatt = acts.getPreatt() + l * B * NH * T * T;
            int l_att = acts.getAtt() + l * B * NH * T * T;
            int l_attproj = acts.getAttproj() + l * B * T * C;
            int l_residual2 = acts.getResidual2() + l * B * T * C;
            int l_ln2 = acts.getLn2() + l * B * T * C;
            int l_ln2_mean = acts.getLn2Mean() + l * B * T;
            int l_ln2_rstd = acts.getLn2Rstd() + l * B * T;
            int l_fch = acts.getFch() + l * B * T * 4 * C;
            int l_fch_gelu = acts.getFchGelu() + l * B * T * 4 * C;
            int l_fcproj = acts.getFcproj() + l * B * T * C;
            int l_residual3 = acts.getResidual3() + l * B * T * C;
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);//checked 1
            matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);//checked 1
            attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);//checked 1
            matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
            residual_forward(l_residual2, residual, l_attproj, B * T * C);//checked 1
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
            matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
            gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);//checked 1
            matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
            residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
        }
        residual = acts.getResidual3() + (L - 1) * B * T * C; // last residual is in residual3
        layernorm_forward(acts.getLnf(), acts.getLnfMean(), acts.getLnfRstd(), residual,
                params.getLnfw(), params.getLnfb(), B, T, C);
        matmul_forward(acts.getLogits(), acts.getLnf(), params.getWte(), Integer.MIN_VALUE, B, T, C, V);
        softmax_forward(acts.getProbs(), acts.getLogits(), B, T, V);
        // also forward the cross-entropy loss function if we have the targets
        if (loader.targetsPresent()) {
            crossentropy_forward(acts.getLosses(), acts.getProbs(), loader, B, T, V);
            // for convenience also evaluate the mean loss
            float mean_loss = 0.0f;
            for (int i = 0; i < B * T; i++) {
                mean_loss += acts.getMem(acts.getLosses() + i);
                Assert.nonNan(mean_loss);
            }
            mean_loss /= B * T;
            this.mean_loss = mean_loss;
            //log.info("f=={} mean_loss == {}", gpt2_forward_counter_layer.get(), mean_loss);
            Assert.nonNan(this.mean_loss);
        } else {
            // if we don't have targets, we don't have a loss
            this.mean_loss = -1.0f;
        }
    }

    public void gpt2_backward() {
        if (mean_loss == -1.0f) {
            log.info("Error: must forward with targets before backward\n");
            System.exit(1);
        }
        // convenience shortcuts
        int B = batch_size;
        int T = seq_len;
        int V = config.vocab_size;
        int L = config.num_layers;
        int NH = config.num_heads;
        int C = config.channels;

        if (this.grads == null) {
            this.grads = new ParameterTensors(config);
            this.grads_acts = new ActivationTensors(config, B, T);
        }
        // backward pass: go in the reverse order of the forward pass, and call backward() functions
        //ParameterTensors params = params; // for brevity
        //ParameterTensors grads = model->grads;
        //ActivationTensors acts = model->acts;
        //ActivationTensors grads_acts = model->grads_acts;

        // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
        // technically this is a small, inline backward() pass of calculating
        // total, final loss as the mean over all losses over all (B,T) positions in the batch
        float dloss_mean = 1.0f / (B * T);
        for (int i = 0; i < B * T; i++) {
            grads_acts.setLosses(i, dloss_mean);
        }
        crossentropy_softmax_backward(grads_acts.getLogits(), grads_acts.getLosses(), acts.getProbs(), loader, B, T, V);
        matmul_backward(grads_acts.getLnf(), grads.getWte(), Integer.MIN_VALUE, grads_acts.getLogits(), acts.getLnf(), params.wte, B, T, C, V);

        // last layer's residual
        int residual = acts.getResidual3() + (L - 1) * B * T * C;
        // write to last layer's residual
        int dresidual = grads_acts.getResidual3() + (L - 1) * B * T * C;

        layernorm_backward(dresidual, grads.getLnfw(), grads.getLnfb(), grads_acts.getLnf(), residual, params.getLnfw(), acts.getLnfMean(), acts.getLnfRstd(), B, T, C);

        int l_ln1 = 0;
        for (int l = L - 1; l >= 0; l--) {
            long layerCount = gpt2_backward_counter_layer.incrementAndGet();
            float residualTest;
            float dresidualTest;
            if (l == 0) {
                residual = acts.getEncoded();
                dresidual = grads_acts.getEncoded();
                residualTest = acts.getMem(residual);
                dresidualTest = grads_acts.getMem(dresidual);
            } else {
                residual = acts.getResidual3() + (l - 1) * B * T * C;
                dresidual = grads_acts.getResidual3() + (l - 1) * B * T * C;
                residualTest = acts.getMem(residual);
                dresidualTest = grads_acts.getMem(dresidual);
            }
            //System.out.printf("b==%d f==%d residual == %f dlresidual == %f\n", layerCount, gpt2_forward_counter_layer.get(), residualTest, dresidualTest);
            // get the pointers of the weights for this layer
            int l_ln1w = params.getLn1w() + l * C;
            int l_qkvw = params.getQkvw() + l * 3 * C * C;
            int l_attprojw = params.getAttprojw() + l * C * C;
            int l_ln2w = params.getLn2w() + l * C;
            int l_fcw = params.getFcw() + l * 4 * C * C;
            int l_fcprojw = params.getFcprojw() + l * C * 4 * C;
            // get the pointers of the gradients of the weights for this layer
            int dl_ln1w = grads.getLn1w() + l * C;
            int dl_ln1b = grads.getLn1b() + l * C;
            int dl_qkvw = grads.getQkvw() + l * 3 * C * C;
            int dl_qkvb = grads.getQkvb() + l * 3 * C;
            int dl_attprojw = grads.getAttprojw() + l * C * C;
            int dl_attprojb = grads.getAttprojb() + l * C;
            int dl_ln2w = grads.getLn2w() + l * C;
            int dl_ln2b = grads.getLn2b() + l * C;
            int dl_fcw = grads.getFcw() + l * 4 * C * C;
            int dl_fcb = grads.getFcb() + l * 4 * C;
            int dl_fcprojw = grads.getFcprojw() + l * C * 4 * C;
            int dl_fcprojb = grads.getFcprojb() + l * C;
            // get the pointers of the activations for this layer
            l_ln1 = acts.getLn1() + l * B * T * C;
            int l_ln1_mean = acts.getLn1Mean() + l * B * T;
            int l_ln1_rstd = acts.getLn1Rstd() + l * B * T;
            int l_qkv = acts.getQkv() + l * B * T * 3 * C;
            int l_atty = acts.getAtty() + l * B * T * C;
            int l_att = acts.getAtt() + l * B * NH * T * T;
            int l_residual2 = acts.getResidual2() + l * B * T * C;
            int l_ln2 = acts.getLn2() + l * B * T * C;
            int l_ln2_mean = acts.getLn2Mean() + l * B * T;
            int l_ln2_rstd = acts.getLn2Rstd() + l * B * T;
            int l_fch = acts.getFch() + l * B * T * 4 * C;
            int l_fch_gelu = acts.getFchGelu() + l * B * T * 4 * C;
            // get the pointers of the gradients of the activations for this layer
            int dl_ln1 = grads_acts.getLn1() + l * B * T * C;
            int dl_qkv = grads_acts.getQkv() + l * B * T * 3 * C;
            int dl_atty = grads_acts.getAtty() + l * B * T * C;
            int dl_preatt = grads_acts.getPreatt() + l * B * NH * T * T;
            int dl_att = grads_acts.getAtt() + l * B * NH * T * T;
            int dl_attproj = grads_acts.getAttproj() + l * B * T * C;
            int dl_residual2 = grads_acts.getResidual2() + l * B * T * C;
            int dl_ln2 = grads_acts.getLn2() + l * B * T * C;
            int dl_fch = grads_acts.getFch() + l * B * T * 4 * C;
            int dl_fch_gelu = grads_acts.getFchGelu() + l * B * T * 4 * C;
            int dl_fcproj = grads_acts.getFcproj() + l * B * T * C;
            int dl_residual3 = grads_acts.getResidual3() + l * B * T * C;

            if (grads_acts.getMem(dl_residual3) > 1f) {
                System.out.printf("dl_residual == %f", grads_acts.getMem(dl_residual3));
                throw new IllegalStateException("This should never happen");
            }

            // backprop this layer
            residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C);// checked 1

            matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C);//checked
            gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C);
            matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C);
            layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean,
                    l_ln2_rstd, B, T, C);
            residual_backward(dresidual, dl_attproj, dl_residual2, B * T * C);
            matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C,
                    C);
            attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
            matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C);
            layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean,
                    l_ln1_rstd, B, T, C);
        }
        encoder_backward(grads.wte, grads.getWpe(), grads_acts.getEncoded(), loader, B, T, C);
        return;
    }

    // acts    , acts
    private void crossentropy_forward(int losses, int probs, DataLoader targets, int B, int T,
                                      int V) {
        //void crossentropy_forward(float* losses,  float* probs, int* targets,int B, int T, int V) {
        // output: losses is (B,T) of the individual losses at each position
        // input: probs are (B,T,V) of the probabilities
        // input: targets is (B,T) of integers giving the correct index in logits
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                // loss = -log(probs[target])
                int probs_bt = probs + (b * T * V + t * V);//acts
                int pos = b * T + t;
                int ix = targets.getTargets(pos);

                float tmp = acts.getMem(probs_bt + ix);
                float neLog = (float) -Math.log(tmp);
                acts.setMem(losses + pos, neLog);

            }
        }

    }

    /*
    probs == acts
    logits == acts
     */
    private void softmax_forward(int probs, int logits, int B, int T, int V) {
        // output: probs are (B,T,V) of the probabilities (sums to 1.0 in each b,t position)
        // input: logits is (B,T,V) of the unnormalized log probabilities
        //#pragma omp parallel for collapse(2)
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                // probs <- softmax(logits)
                int logits_bt = logits + (b * T * V + t * V);
                int probs_bt = probs + (b * T * V + t * V);

                // maxval is only calculated and subtracted for numerical stability
                float maxval = -10000.0f; // TODO something better
                for (int i = 0; i < V; i++) {
                    if (acts.getMem(logits_bt + i) > maxval) {
                        //maxval = logits_bt[i];
                        maxval = acts.getMem(logits_bt + i);
                    }
                }
                float sum = 0.0f;
                for (int i = 0; i < V; i++) {
                    float tmp = (float) Math.exp(acts.getMem(logits_bt + i) - maxval);
                    //probs_bt[i] = Math.exp(logits_bt[i] - maxval);
                    acts.setMem(probs_bt + i, tmp);
                    //sum += probs_bt[i];
                    sum = sum + tmp;
                }
                for (int i = 0; i < V; i++) {
                    //probs_bt[i] /= sum;
                    //probs_bt[i] = probs_bt[i] / sum;
                    float tmp = acts.getMem(probs_bt + i);
                    float val = tmp / sum;
                    acts.setMem(probs_bt + i, val);
                }
            }
        }
    }

  /*
  out ==
  inp ==
   */

    private void gelu_forward(int out, int inp, int N) {
        //#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
        // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
        for (int i = 0; i < N; i++) {
            float x = acts.getMem(inp + i);
            float cube = 0.044715f * x * x * x;
            float tmp = (float) (0.5f * x * (1.0f + Math.tanh(GELU_SCALING_FACTOR * (x + cube))));
            acts.setMem(out + i, tmp);
        }
    }


  /*
  out == acts
  inp1 == acts
  inp2 == acts
   */

    private void residual_forward(int out, int inp1, int inp2, int N) {
        for (int i = 0; i < N; i++) {
            //out[i] = inp1[i] + inp2[i];
            float inp1_i = acts.getMem(inp1 + i);
            float inp2_i = acts.getMem(inp2 + i);
            float result = inp1_i + inp2_i;
//      if(i < 5) {
//        System.out.printf("inp1_i == %18.18f  inp2_id == %18.18f result = %18.18f\n", inp1_i,
//            inp2_i, result);
//      }
            acts.setMem(out + i, result);
        }
    }

    /*
      out == acts
      preatt == acts
      att == acts
      inp == acts
     */
    //void attention_forward(float* out, float* preatt, float* att, float* inp, int B, int T, int C, int NH)
    private void attention_forward(int out, int preatt, int att, int inp, int B, int T, int C,
                                   int NH) {
        // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
        // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
        // that holds the pre-attention and post-attention scores (used in backward)
        // output is (B, T, C)
        // attention is the only layer that mixes information across time
        // every other operation is applied at every (b,t) position independently
        // (and of course, no layer mixes information across batch)
        int C3 = C * 3;
        int hs = C / NH; // head size
        float scale = (float) (1.0 / Math.sqrt(hs));

        //#pragma omp parallel for collapse(3)
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int h = 0; h < NH; h++) {
                    final int query_t = inp + b * T * C3 + t * C3 + h * hs;//acts
                    final int preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;//acts
                    final int att_bth = att + b * NH * T * T + h * T * T + t * T;

                    // pass 1: calculate query dot key and maxval
                    float maxval = -10000.0f; // TODO something better
                    for (int t2 = 0; t2 <= t; t2++) {
                        int key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                        // (query_t) dot (key_t2)
                        float val = 0.0f;
                        for (int i = 0; i < hs; i++) {
                            val += acts.getMem(query_t + i)
                                    * acts.getMem(key_t2 + i);
                        }
                        val *= scale;
                        if (val > maxval) {
                            maxval = val;
                        }
                        acts.setMem(preatt_bth + t2, val);
                    }

                    // pass 2: calculate the exp and keep track of sum
                    // maxval is being calculated and subtracted only for numerical stability
                    float expsum = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        //float expv = expf(preatt_bth[t2] - maxval);
                        float expv = (float) Math.exp(acts.getMem(preatt_bth + t2) - maxval);
                        expsum += expv;
                        acts.setMem(att_bth + t2, expv);
                    }
                    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                    // pass 3: normalize to get the softmax
                    //float tmp = acts.getMem(att_bth);
                    for (int t2 = 0; t2 < T; t2++) {
                        if (t2 <= t) {
                            //att_bth[t2] *= expsum_inv;
                            float tmp = acts.getMem(att_bth + t2) * expsum_inv;
                            acts.setMem(att_bth + t2, tmp);
                        } else {
                            // causal attention mask. not strictly necessary to set to zero here
                            // only doing this explicitly for debugging and checking to PyTorch
                            //att_bth[t2] = 0.0f;
                            acts.setMem(att_bth + t2, 0.0f);
                        }
                    }

                    // pass 4: accumulate weighted values into the output of attention
                    int out_bth = out + b * T * C + t * C + h * hs;
                    for (int i = 0; i < hs; i++) {
                        acts.setMem(out_bth + i, 0.0f);
                    }
                    for (int t2 = 0; t2 <= t; t2++) {
                        int value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
                        float att_btht2 = acts.getMem(att_bth + t2);
                        for (int i = 0; i < hs; i++) {
                            //out_bth[i] += att_btht2 * value_t2[i];
                            //out_bth[i] = out_bth[i] att_btht2 * value_t2[i];

                            float tmp = acts.getMem(out_bth + i)
                                    + att_btht2
                                    * acts.getMem(value_t2 + i);
                            acts.setMem(out_bth + i, tmp);
                        }
                    }
                }
            }
        }

    }

    /*
     out == acts
     inp == acts
     weight == params
     bias == params
     */
    //
    private void matmul_forward(int out, int inp, int weight, int bias, int B, int T, int C, int OC) {

        /**
         * Loop unrolling
         * https://stackoverflow.com/questions/28482833/understanding-the-collapse-clause-in-openmp
         *
         */
        final int btMax = B * T;
        IntStream.range(0, btMax) // This is probably not the fastest way to thread this loop
                .parallel()
                .forEach(bt -> {
                    int b = bt / T;
                    int t = bt % T;
                    final int out_bt = out + (b * T * OC + t * OC);//acts
                    final int inp_bt = inp + (b * T * C + t * C);//acts
                    for (int o = 0; o < OC; o++) {
                        final int oo = o;
                        float val = 0.0f;
                        if (bias != Integer.MIN_VALUE) {
                            val = params.getMem(bias + oo);
                        }
                        int wrow = weight + oo * C;
                        for (int i = 0; i < C; i++) {
                            float inp_bt_i = acts.getMem(inp_bt + i);
                            float wrow_i = params.getMem(wrow + i);
                            val += inp_bt_i * wrow_i;
                        }
                        acts.setMem(out_bt + oo, val);
                    }
                });
    }

    private void matmul_forward_slow(int out, int inp, int weight, int bias, int B, int T, int C, int OC) {

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                final int out_bt = out + (b * T * OC + t * OC);//acts
                final int inp_bt = inp + (b * T * C + t * C);//acts
                for (int o = 0; o < OC; o++) {
                    final int oo = o;
                    float val = 0.0f;
                    if (bias != Integer.MIN_VALUE) {
                        val = params.getMem(bias + oo);
                    }
                    int wrow = weight + oo * C;
                    for (int i = 0; i < C; i++) {
                        float inp_bt_i = acts.getMem(inp_bt + i);
                        float wrow_i = params.getMem(wrow + i);
                        val += inp_bt_i * wrow_i;
                    }
                    acts.setMem(out_bt + oo, val);

                }
            }
        }
    }


    // out == acts
    // mean == acts
    // rstd == acts
    // inp == acts
    // weight == params
    // bias == params
    private void layernorm_forward(int out, int mean, int rstd, int inp, int weight, int bias, int B,
                                   int T, int C) {
        // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        // both inp and out are (B,T,C) of the activations
        // mean and rstd are (B,T) buffers, to be used later in backward pass
        // at each position (b,t) of the input, the C-dimensional vector
        // of activations gets normalized, then scaled and shifted
        float eps = 1e-5f;
        int counter = 0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                // seek to the input position inp[b,t,:]
                int x = inp + (b * T * C + t * C);//acts
                counter++;
                // calculate the mean
                float m = 0.0f;
                for (int i = 0; i < C; i++) {
                    //m += x[i];
                    float x_i = acts.getMem(x + i);
                    m += x_i;

//          if(b == 0 && t == 0 && i == 5) {
//            int in = Float.floatToIntBits(x_i);
//            System.out.printf("x[i] %1.17f %d %s\n", x_i, in, Integer.toBinaryString(in));
//          }
                }
                m = m / C;
                //System.out.printf("%s\n", Integer.toBinaryString(Float.floatToIntBits(m)));
                // calculate the variance (without any bias correction)
                float v = 0.0f;
                for (int i = 0; i < C; i++) {
                    //float xshift = x[i] - m;
                    float xshift = acts.getMem(x + i) - m;
                    v += xshift * xshift;
                }
                v = v / C;
                // calculate the rstd (reciprocal standard deviation)
                //float s = 1.0f / sqrtf(v + eps);
                float s = (float) (1.0f / Math.sqrt(v + eps));
                // seek to the output position in out[b,t,:]
                //float*out_bt = out + b * T * C + t * C;
                int out_bt = out + (b * T * C + t * C);//acts

                for (int i = 0; i < C; i++) {
                    float n = (s * (acts.getMem(x + i) - m)); // normalize

                    float o = n * params.getMem(weight + i)
                            + params.getMem(bias + i); // scale and shift

                    acts.setMem(out_bt + i, o); // write
                }
                // cache the mean and rstd for the backward pass later
                acts.setMem(mean + (b * T + t), m);
                acts.setMem(rstd + (b * T + t), s);
//        if(b == 0 && t == 0) {
//          System.out.printf("m %1.17f %s\n", m, Integer.toBinaryString(Float.floatToIntBits(m)));
//          System.out.printf("s %1.17f %s\n", s, Integer.toBinaryString(Float.floatToIntBits(s)));
//        }
            }
        }
    }

    // out == acts
    // wte == params
    // wpe == params
    private void encoder_forward(int out, DataLoader inp, int wte, int wpe, int B, int T, int C) {
        // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
        // inp is (B,T) of integers, holding the token ids at each (b,t) position
        // wte is (V,C) of token embeddings, short for "weight token embeddings"
        // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                // seek to the output position in out[b,t,:]
                //float* out_bt = out + b * T * C + t * C;
                int out_bt = out + b * T * C + t * C;//acts

                // get the index of the token at inp[b, t]
                int ix = inp.getInputs(b * T + t);

                // seek to the position in wte corresponding to the token
                //float* wte_ix = wte + ix * C;
                int wte_ix = wte + ix * C;//params

                // seek to the position in wpe corresponding to the position
                //float* wpe_t = wpe + t * C;
                int wpe_t = wpe + t * C;//params

                // add the two vectors and store the result in out[b,t,:]
                for (int i = 0; i < C; i++) {
                    //out_bt[i] = wte_ix[i] + wpe_t[i];
                    float wte_ix_i = params.getMem(wte_ix + i);
                    float wpe_t_i = params.getMem(wpe_t + i);

                    float tmp = wte_ix_i + wpe_t_i;
                    acts.setMem(out_bt + i, tmp);
//          if(b < 1 && t < 1 && i < 2) {
//            System.out.printf("tmp == %1.17f wte_ix_i == %1.17f wpe_ix_i == %1.17f\n", tmp, wte_ix_i, wpe_t_i);
//          }
                }
            }
        }
    }


    public int getNumParams() {
        return num_parameters;
    }
}
