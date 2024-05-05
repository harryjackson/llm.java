package org.hjackson.llm;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.concurrent.atomic.AtomicLong;

public final class DataLoader {
    private final int BT1;
    // hyperparameters
    public final int B; // batch size
    public final int T; // sequence length
    private final String type;
    // input handling and its state
    public RandomAccessFile tokens_file;
    public long file_size;
    public long current_position;
    // output memory
    public final int[] batch;
    private final int[] cache;
    private boolean workOnCache = false;
    // convenience variables
    public int num_batches;
    private boolean targetsPresent  = false;
    public DataLoader(String filename, final int B, final int T, String type, boolean targetsPresent) throws Exception {
        this(B, T, type);
        this.targetsPresent = true;
        // open the input file for reading
        this.tokens_file = new RandomAccessFile(new File(filename), "r");
        this.file_size = this.tokens_file.length();
        if (this.file_size < (B * T + 1) * 4) {
            throw new Exception("Error: file size is too small for the batch size and sequence length");
        }
        this.current_position = 0; // start at the beginning
        this.num_batches = (int) ( this.file_size / (B * T * 4));
    }

    public DataLoader(int[] genTokens, final int B, final int T, String type, boolean targetsPresent) {
        this(B, T, type);
        this.targetsPresent = targetsPresent;
        // allocate space for B*T + 1 integers to store the inputs and targets
        System.arraycopy(genTokens, 0, batch, 0, genTokens.length);
    }

    public DataLoader(int B, int T, String type) {
        this.B = B;
        this.T = T;
        this.type = type;
        this.BT1 = B*T+1;
        // allocate space for B*T + 1 integers to store the inputs and targets
        this.batch = new int[BT1];
        this.cache = new int[BT1];
    }

    public void dataloader_reset() {
        this.current_position = 0; // start at the beginning
    }

    public void dataloader_next_batch() throws IOException {
        // if we are at the end of the file, loop back to the beginning
        if (this.current_position + (BT1) * 4 > this.file_size) {
            this.current_position = 0;
        }
        // read the B*T+1 integers from the file into batch
        tokens_file.seek(this.current_position);
        for(int i = 0; i < BT1; i++) {
            this.batch[i] = Integer.reverseBytes(tokens_file.readInt());
        }
        cacheInputs();
        this.current_position += (long) this.B * this.T * 4;
        long current_position = tokens_file.getChannel().position();
        if (current_position != this.current_position + 4) {
            throw new IOException("Invalid file operation.");
        }
    }
    public int getInputs(int i) {
        int res = batch[i];
        if(workOnCache) {
            res = cache[i];
        }
        return res;
    }
    public int getTargets(int i) {
        int res = batch[i + 1];
        if(workOnCache) {
            res = cache[i + 1];
        }
        return res;
    }

    public boolean targetsPresent() {
        return targetsPresent;
    }
    public void cacheInputs() {
        System.arraycopy(batch, 0, cache, 0, batch.length);
    }
    public void setWorkOnCache(boolean woncache) {
        workOnCache = woncache;
    }

}
