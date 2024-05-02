package org.hjackson.llm;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.time.Instant;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class GPT2Test {
    private static final Logger log = LoggerFactory.getLogger(GPT2Test.class);
    float epsilon = 1e-2f;
    private int B;
    private int T;
    private GPT2 model;
    private long file_size;
    private Arena memoryArena;
    private MemorySegment mappedFile;
    private ParameterTensors expected_grads;
    private int C;
    private int V;
    private int maxT;
    private int L;
    private int[] state_header;
    private byte[] mem;
    private int cpos;
    private byte[] fbuf = new byte[4];
    private int n;
    private int Vp;

    @BeforeEach
    void setUp() throws Exception {

        String stateFile = "gpt2_124M_debug_state.bin";
        String checkpoint_path = "gpt2_124M.bin";
        model = new GPT2(checkpoint_path);

        this.C = model.config.channels;
        this.V = model.config.vocab_size;
        this.Vp = model.config.padded_vocab_size;
        this.maxT = model.config.max_seq_len;
        this.L = model.config.num_layers;

        state_header = new int[256];
        RandomAccessFile state_file = new RandomAccessFile(new File(stateFile), "r");
        /**
         * Reading the entire file into mem is orders of magnitude faster than looping over the file and reading it in.
         */
        mem = new byte[(int) state_file.length()];
        cpos = 0;
        state_file.read(mem);
        state_file.close();

        for (int i = 0; i < 256; i++) {
            fbuf[0] = mem[cpos];
            fbuf[1] = mem[cpos + 1];
            fbuf[2] = mem[cpos + 2];
            fbuf[3] = mem[cpos + 3];
            state_header[i] = ByteBuffer.wrap(fbuf).order(ByteOrder.nativeOrder()).getInt();
            cpos += 4;
        }
        Assertions.assertEquals(20240327, state_header[0], "Bad magic state file");
        Assertions.assertEquals(2, state_header[1], "Bad version in state file");
        B = this.state_header[2]; // batch size, e.g. 4
        T = this.state_header[3];
        log.info("[State]");
        log.info("batch_size: {}", B);
        log.info("seq_len: {}", T);
    }

    @Test
    void gpt2_build_from_checkpoint() throws IOException {
        Assertions.assertEquals(20240327, state_header[0], "Bad magic state file");
        Assertions.assertEquals(2, state_header[1], "Bad version in state file");
        int[] x = new int[B * T];
        int[] y = new int[B * T];

        for (int i = 0; i < x.length; i++) {
            n = i;
            fbuf[0] = mem[cpos];
            fbuf[1] = mem[cpos + 1];
            fbuf[2] = mem[cpos + 2];
            fbuf[3] = mem[cpos + 3];
            x[i] = ByteBuffer.wrap(fbuf).order(ByteOrder.LITTLE_ENDIAN).getInt();
            //System.out.printf("%d %d\n", n, x[i]);
            cpos += 4;
        }
        DataLoader loader = new DataLoader(x, B, T);

        for (int i = 0; i < y.length; i++) {
            n++;
            fbuf[0] = mem[cpos];
            fbuf[1] = mem[cpos + 1];
            fbuf[2] = mem[cpos + 2];
            fbuf[3] = mem[cpos + 3];
            y[i] = ByteBuffer.wrap(fbuf).order(ByteOrder.LITTLE_ENDIAN).getInt();
            cpos += 4;
        }

        float expected_loss = 0.0f;
        final int num_params = model.getNumParams();

        int btv = B * T * V;
        //ByteBuffer buf = ByteBuffer.
        fbuf = new byte[4];
        float[] expected_logits = new float[btv];
        log.info("reading expected_logits");
        for (int i = 0; i < btv; i++) {
            n++;
            fbuf[0] = mem[cpos];
            fbuf[1] = mem[cpos + 1];
            fbuf[2] = mem[cpos + 2];
            fbuf[3] = mem[cpos + 3];
            float f = ByteBuffer.wrap(fbuf).order(ByteOrder.LITTLE_ENDIAN).getFloat();
            //System.out.printf("%d %1.17f\n", n, f);
            expected_logits[i] = f;
            cpos += 4;
        }

        fbuf[0] = mem[cpos];
        fbuf[1] = mem[cpos + 1];
        fbuf[2] = mem[cpos + 2];
        fbuf[3] = mem[cpos + 3];
        cpos += 4;
        n++;
        expected_loss = ByteBuffer.wrap(fbuf).order(ByteOrder.LITTLE_ENDIAN).getFloat();

        float[] expected_grads_memory = new float[num_params];
        log.info("reading expected_grads_memory");
        for (int i = 0; i < num_params; i++) {
            n++;
            fbuf[0] = mem[cpos];
            fbuf[1] = mem[cpos + 1];
            fbuf[2] = mem[cpos + 2];
            fbuf[3] = mem[cpos + 3];
            float f = ByteBuffer.wrap(fbuf).order(ByteOrder.LITTLE_ENDIAN).getFloat();
            //System.out.printf("%d %1.17f\n", n, f);
            expected_grads_memory[i] = f;
            cpos += 4;
        }
        log.info("cpos == {}", cpos);
        Assertions.assertEquals(549369860, cpos);

    /* expected_logits[0]    == -43.43161774
       expected_loss         == 5.27000856
       expected_grads_memory == -0.00231974 */
        log.info("expected_logits[0] == {} length == {}", expected_logits[0], expected_logits.length);
        log.info("expected_loss            == {}", expected_loss);
        log.info("expected_grads_memory[0] == {} length == {}", expected_grads_memory[0], expected_grads_memory.length);

        // overall OK signal for the test
        boolean allok = true;

        // expected losses are as follows, from Python
        float[] expected_losses = {
                5.270007133483887f,
                4.059706687927246f,
                3.3751230239868164f,
                2.8007826805114746f,
                2.315382242202759f,
                1.8490285873413086f,
                1.3946564197540283f,
                0.9991465210914612f,
                0.6240804195404053f,
                0.37651097774505615f
        };

        // let's do 10 training iterations, following the pytorch code
        float[] losses = new float[10];
        for (int step = 0; step < 10; step++) {
            Instant start = Instant.now();
            model.gpt2_forward(loader, B, T);
            model.gpt2_zero_grad();
            model.gpt2_backward();
            Instant end = Instant.now();
            log.info("Duration: {}", Duration.between(end, start));
            ActivationTensors acts = model.acts;

            if (step == 0) {
                // error checking at step 0 for reference activations/gradients
                // at this point, target should be equal to expected_logits, let's compare
                boolean logits_ok = true;
                int calculated_logits = acts.getLogits();
                float max_diff = 0.0f;
                for (int bt = 0; bt < B * T; bt++) {

                    for (int v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
                        int i = bt * Vp + v; // linearized index, using Vp
                        if (i < 10) {
                            System.out.printf("%1.10f, %1.10f\n", expected_logits[i], model.acts.mem[calculated_logits + i]);
                        }
                        float diff = Math.abs(expected_logits[bt * V + v] - model.acts.mem[calculated_logits + i]);
                        max_diff = Math.max(max_diff, diff);
                        if (diff >= 1e-2f) {
                            System.out.printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                            System.out.printf("%1.10f %1.10f\n", expected_logits[bt * V + v], model.acts.mem[calculated_logits + i]);
                            logits_ok = false;
                            bt = B * T; // to break out of both loops
                            break;
                        }
                    }
                }

                if (!logits_ok) {
                    log.error("Logits not ok, exiting");
                    System.exit(1);
                }

                log.info("OK (LOGITS)");
                allok = allok && logits_ok;

                // compare the achieved loss
                if (Math.abs(model.mean_loss - expected_loss) >= epsilon) {
                    log.info("LOSS MISMATCH: {} {}", model.mean_loss, expected_loss);
                    allok = false;
                } else {
                    log.info("LOSS OK: {} {}", model.mean_loss, expected_loss);
                }

                // finally check all the gradients
                boolean[] gradoks = new boolean[16];
                ParameterTensors grads = model.grads;

                gradoks[0] = check_tensor(grads.getWte(), expected_grads_memory, V * C, "dwte");
                //gradoks[1] = check_tensor(grads.wpe, expected_grads.wpe, maxT*C, "dwpe");
//        gradoks[2] = check_tensor(grads.ln1w, expected_grads.ln1w, L*C, "dln1w");
//        gradoks[3] = check_tensor(grads.ln1b, expected_grads.ln1b, L*C, "dln1b");
//        gradoks[4] = check_tensor(grads.qkvw, expected_grads.qkvw, L*3*C*C, "dqkvw");
//        gradoks[5] = check_tensor(grads.qkvb, expected_grads.qkvb, L*3*C, "dqkvb");
//        gradoks[6] = check_tensor(grads.attprojw, expected_grads.attprojw, L*C*C, "dattprojw");
//        gradoks[7] = check_tensor(grads.attprojb, expected_grads.attprojb, L*C, "dattprojb");
//        gradoks[8] = check_tensor(grads.ln2w, expected_grads.ln2w, L*C, "dln2w");
//        gradoks[9] = check_tensor(grads.ln2b, expected_grads.ln2b, L*C, "dln2b");
//        gradoks[10] = check_tensor(grads.fcw, expected_grads.fcw, L*4*C*C, "dfcw");
//        gradoks[11] = check_tensor(grads.fcb, expected_grads.fcb, L*4*C, "dfcb");
//        gradoks[12] = check_tensor(grads.fcprojw, expected_grads.fcprojw, L*C*4*C, "dfcprojw");
//        gradoks[13] = check_tensor(grads.fcprojb, expected_grads.fcprojb, L*C, "dfcprojb");
//        gradoks[14] = check_tensor(grads.lnfw, expected_grads.lnfw, C, "dlnfw");
//        gradoks[15] = check_tensor(grads.lnfb, expected_grads.lnfb, C, "dlnfb");
//        for (int i = 0; i < 16; i++) {
//          allok = allok && gradoks[i];
//        }
            }
        }


        // compare
        for (int i = 0; i < 10; i++) {
            if (Math.abs(losses[i] - expected_losses[i]) >= 1e-2) {
                log.info("LOSS MISMATCH AT STEP {}: {} {}", i, losses[i], expected_losses[i]);
                allok = false;
            } else {
                log.info("loss ok at step {}: {} {}", i, losses[i], expected_losses[i]);
            }
        }
        log.info("overall okay: {}", allok);
    }


    boolean check_tensor(int start, float[] b, int n, String label) {
        int print_upto = 5;
        boolean ok = true;
        float maxdiff = 0.0f;
        float tol = 2e-2f;
        log.info("{}", label);
        for (int i = 0; i < n; i++) {
            float diff = Math.abs(model.grads.mem[start + i] - b[i]);
            ok = ok && (diff <= tol);
            if (diff > maxdiff) {
                maxdiff = diff;
            }
            // for the first few elements of each tensor, pretty print
            // the actual numbers, so we can do a visual, qualitative proof/assessment
            if (i < print_upto) {
                if (diff <= tol) {
                    if (i < print_upto) {
                        System.out.printf("OK ");
                    }
                } else {
                    if (i < print_upto) {
                        System.out.printf("NOT OK ");
                    }
                }
                System.out.printf("%f %f\n", model.grads.mem[start + i], b[i]);
            }
        }
        // print the final result for this tensor
        if (ok) {
            System.out.printf("TENSOR OK, maxdiff = %e\n", maxdiff);
        } else {
            System.out.printf("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
        }
        return ok;
    }

    @Test
    public void randomTest() {
        float test = GPT2.random_f32(1337l);
        System.out.printf("%1.17f", test);
        Assertions.assertEquals(0.23031723499298096f, test);
    }

}