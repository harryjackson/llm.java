package org.hjackson.llm;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Llm {
    private static String tiny_stories_train = "data/TinyStories_train.bin";
    private static String tiny_stories_val = "data/TinyStories_val.bin";
    private static String tiny_shakespeare_train = "data/tiny_shakespeare_train.bin";
    private static String tiny_shakespeare_val = "data/tiny_shakespeare_val.bin";

    //private static final Random random = new Random(RNG_STATE);

    public static void main(String[] args) throws Exception {
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        long xmx = memoryBean.getHeapMemoryUsage().getMax();
        System.out.printf("Max Mem %d\n", xmx);
        Llm llm = new Llm();
        if(xmx < 8589934590L) {
            throw new IllegalStateException("-Xmx needs to be at least -Xmx8192m");
        }
        if(!llm.getClass().desiredAssertionStatus()) {
            System.out.printf("Assertions are turned off, if editing the code I strongly recommend them to be on\n");
        }

        System.out.printf("Hello and welcome!\n");
        GPT2 model = new GPT2("gpt2_124M.bin");
        System.out.printf("wte[0] == %f\n", model.params.mem[0]);
        // build the DataLoaders from tokens files. for now use tiny_shakespeare if
        // available, else tiny_stories
        String train_tokens = Files.exists(Paths.get(tiny_shakespeare_train)) ? tiny_shakespeare_train : tiny_stories_train;
        String val_tokens = Files.exists(Paths.get(tiny_shakespeare_val)) ? tiny_shakespeare_val : tiny_stories_val;
        System.out.printf("Training with %s using values %s\n", train_tokens, val_tokens);
        final int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
        final int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT,
        // which is 1024 for GPT-2
        DataLoader train_loader = new DataLoader(train_tokens, B, T, "train", true);
        System.out.printf("train dataset num_batches: %s\n", train_loader.num_batches);
        DataLoader val_loader = new DataLoader(val_tokens, B, T, "val", true);
        System.out.printf("val dataset num_batches: %s\n",  val_loader.num_batches);
        final int val_num_batches = 5;
        Tokenizer tokenizer = new Tokenizer("gpt2_tokenizer.bin");
        // some memory for generating samples from the model
        // int gen_max_length = 64;
        // during inference step we'll generate sequences of this many tokens
        final int[] gen_tokens = new int[B * T];
        final int genT = 64;
        // train
        long start, end;
        for(int step = 0; step <= 40; step++) {
            // once in a while estimate the validation loss
            if (step % 10 == 0) {
                float val_loss = 0.0f;
                val_loader.dataloader_reset();
                for (int i = 0; i < val_num_batches; i++) {
                    val_loader.dataloader_next_batch();
                    model.gpt2_forward(val_loader, B, T);
                    val_loss += model.mean_loss;
                }
                val_loss /= val_num_batches;
                System.out.printf("step %d val loss %f\n", step , val_loss);
            }
            // once in a while do model inference to print generated text
            if (step > 0 && step % 4 == 0) {
                // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
                for(int i = 0; i < B * T; ++i) {
                    gen_tokens[i] = tokenizer.end_of_text;
                }
                // now sample from the model autoregressively
                System.out.printf("generating:\n---\n");
                for (int t = 1; t < genT; t++) {
                    // note that inference is very wasteful here because for each token
                    // we re-calculate the forward pass for all of (B,T) positions from scratch
                    // but the inference here is just for sanity checking anyway
                    // and we can maybe optimize a bit more later, with careful tests
                    DataLoader genTokenLoader = new DataLoader(gen_tokens, B, T, "gen", false);
                    model.gpt2_forward(genTokenLoader, B, T);
                    // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                    // we're in principle running B "inference streams" in parallel here
                    // but only using position 0
                    // get the Vp-dimensional vector probs[0, t-1, :]
                    int probs = model.acts.getProbs() + (t-1) * model.config.padded_vocab_size;
                    float coin = Random32.random_f32(Random32.RNG_STATE);
                    // note we're only sampling from the first V elements, ignoring padding
                    // (the probabilities in the padded region should be zero anyway)
                    int next_token = model.sample_mult(probs, model.config.vocab_size, coin);
                    //System.out.printf("\nnext_token == %d coin == %f probs[0] == %f rng_state == %d\n", next_token, coin, model.acts.mem[probs], Random32.RNG_STATE);
                    gen_tokens[t] = next_token;
                    // print the generated token, either using the Tokenizer or a fallback
                    if (tokenizer.init_ok == 1) {
                        String token_str = tokenizer.tokenizer_decode(next_token);
                        System.out.printf(token_str);
                    } else {
                        // fall back to printing the token id
                        System.out.printf(String.valueOf(next_token));
                    }
                }
                System.out.printf("\n---\n");
            }
            // do a training step
            start = System.currentTimeMillis();
            train_loader.dataloader_next_batch();
            model.gpt2_forward(train_loader, B, T);
            model.gpt2_zero_grad();
            model.gpt2_backward();
            model.gpt2_update(1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
            end = System.currentTimeMillis(); //clock_gettime(CLOCK_MONOTONIC, &end);
            long time_elapsed_s = end - start;
            System.out.printf("step %d: train loss %f (took %d ms)\n", step, model.mean_loss, time_elapsed_s);
        }
    }
}