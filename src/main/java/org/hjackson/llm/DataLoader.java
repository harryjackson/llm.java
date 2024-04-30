package org.hjackson.llm;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

public final class DataLoader {
    // hyperparameters
    public int B; // batch size
    public int T; // sequence length
    // input handling and its state
    public RandomAccessFile tokens_file;
    public long file_size;
    public long current_position;
    // output memory
    public int[] batch;
    private int[] cache;
    //public int inputs;
    //public int targets;
    // convenience variables
    public int num_batches;
    private boolean targetsPresent  = false;

    public DataLoader(String filename, final int B, final int T) throws Exception {
        this(B, T);
        this.B = B;
        this.T = T;
        targetsPresent = true;

        // open the input file for reading
        this.tokens_file = new RandomAccessFile(new File(filename), "r");
        this.file_size = this.tokens_file.length();

        if (this.file_size < (B * T + 1) * 4) {
            throw new Exception("Error: file size is too small for the batch size and sequence length");
        }
        this.current_position = 0; // start at the beginning

        // this.targets = this.batch + 1; // targets are shifted by one
        this.num_batches = (int) ((long) this.file_size / (B * T * 4));
    }

    public DataLoader(int[] genTokens, final int B, final int T) {
        this(B, T);
        targetsPresent = true;
        // allocate space for B*T + 1 integers to store the inputs and targets
        System.arraycopy(genTokens, 0, batch, 0, genTokens.length);
        cacheInputs();
    }

    public DataLoader(int B, int T) {
        this.B = B;
        this.T = T;
        // allocate space for B*T + 1 integers to store the inputs and targets
        this.batch = new int[B * T + 1];
        this.cache = new int[B * T + 1];
    }

    public void dataloader_reset() {
        this.current_position = 0; // start at the beginning
    }

    public void dataloader_next_batch() throws IOException {
        int B = this.B;
        int T = this.T;
        // if we are at the end of the file, loop back to the beginning
        if (this.current_position + (B * T + 1) * 4 > this.file_size) {
            this.current_position = 0;
        }
        // read the B*T+1 integers from the file into batch
        tokens_file.seek(this.current_position);
        int max = B * T + 1;
        for(int i = 0; i < max; i++) {
            this.batch[i] = Integer.reverseBytes(tokens_file.readInt());
        }
        cacheInputs();
        this.current_position += (long) B * T * 4;
        long current_position = tokens_file.getChannel().position();
        if (current_position != this.current_position + 4) {
            throw new IOException("Invalid file operation.");
        }
    }

    public int getInputs(int i) {
        return batch[i];
    }

    public int getTargets(int i) {
        return batch[i + 1];
    }

    public boolean targetsPresent() {
        return targetsPresent;
    }

    private void cacheInputs() {
        System.arraycopy(batch, 0, cache, 0, batch.length);
    }
}
