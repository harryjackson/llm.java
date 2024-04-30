package org.hjackson.llm;


import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.FloatVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Paths;

public class Llm {
    private static final Logger log = LoggerFactory.getLogger(Llm.class);

    private static final long RNG_STATE = 1337;
    private static String tiny_stories_train = "data/TinyStories_train.bin";
    private static String tiny_stories_val = "data/TinyStories_val.bin";
    private static String tiny_shakespeare_train = "data/tiny_shakespeare_train.bin";
    private static String tiny_shakespeare_val = "data/tiny_shakespeare_val.bin";


    public static void main(String[] args) throws Exception {
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        long xmx = memoryBean.getHeapMemoryUsage().getMax();
        log.info("Max Mem {}", xmx);
        Llm llm = new Llm();
        if(xmx < 8589934590L) {
            throw new IllegalStateException("-Xmx needs to be at least -Xmx8192m");
        }
        if(!llm.getClass().desiredAssertionStatus()) {
            log.warn("Assertions are turned off, if editing the code I strongly recommend them to be on");
        }
        log.info("Hello and welcome!");
        GPT2 model = new GPT2("gpt2_124M.bin");
        log.info("wte[0] == {}", model.params.getMem(0));

        // build the DataLoaders from tokens files. for now use tiny_shakespeare if
        // available, else tiny_stories
        String train_tokens = Files.exists(Paths.get(tiny_shakespeare_train)) ? tiny_shakespeare_train
                : tiny_stories_train;
        String val_tokens = Files.exists(Paths.get(tiny_shakespeare_val)) ? tiny_shakespeare_val : tiny_stories_val;

        final int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
        final int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT,
        // which is 1024 for GPT-2
        DataLoader train_loader = new DataLoader(train_tokens, B, T);
        log.info("train dataset num_batches: {}", train_loader.num_batches + "\n");
        DataLoader val_loader = new DataLoader(val_tokens, B, T);
        log.info("val dataset num_batches: {}",  val_loader.num_batches + "\n");
        int val_num_batches = 10;

        Tokenizer tokenizer = new Tokenizer("gpt2_tokenizer.bin");
        // some memory for generating samples from the model
        //int gen_max_length = 64;
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
                    //log.info("val loss: {} mean_loss: {}", val_loss, model.mean_loss);
                }
                val_loss /= val_num_batches;
                log.info("step {} val loss {}", step , val_loss);
            }

            // once in a while do model inference to print generated text
            if (step > 0 && step % 20 == 0) {
                // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
                for(int i = 0; i < B * T; ++i) {
                    gen_tokens[i] = GPT2.GPT2_EOT;
                }
                // now sample from the model autoregressively
                log.info("generating:\n---\n");
                for (int t = 1; t < genT; t++) {
                    // note that inference is very wasteful here because for each token
                    // we re-calculate the forward pass for all of (B,T) positions from scratch
                    // but the inference here is just for sanity checking anyway
                    // and we can maybe optimize a bit more later, with careful tests
                    DataLoader genTokenLoader = new DataLoader(gen_tokens, B, T);
                    model.gpt2_forward(genTokenLoader, B, T);
                    // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                    // we're in principle running B "inference streams" in parallel here
                    // but only using position 0
                    // get the V-dimensional vector probs[0, t-1, :]
                    int probs = model.acts.getProbs() + (t-1) * model.config.vocab_size;
                    float coin = model.random_f32(RNG_STATE);
                    int next_token = model.sample_mult(probs, model.config.vocab_size, coin);
                    gen_tokens[t] = next_token;
                    // print the generated token, either using the Tokenizer or a fallback
                    if (tokenizer.init_ok == 1) {
                        String token_str = tokenizer.tokenizer_decode(next_token);
                        log.info(token_str);
                    } else {
                        // fall back to printing the token id
                        log.info("{} ", next_token);
                    }
                }
                log.info("\n---\n");
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
            log.info("step {}: train loss {} (took {} ms)\n", step, model.mean_loss, time_elapsed_s);
        }

    }


}