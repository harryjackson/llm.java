package org.hjackson.llm;
import java.nio.IntBuffer;
public final class GPT2Config {
    public int max_seq_len; // max sequence length, e.g. 1024
    public int vocab_size; // vocab size, e.g. 50257
    public int padded_vocab_size; // padded to e.g. %128==0, 50304
    public int num_layers; // number of layers, e.g. 12
    public int num_heads; // number of heads in attention, e.g. 12
    public int channels; // number of channels, e.g. 768
    public GPT2Config(IntBuffer header) {
        this.max_seq_len = header.get(2);//maxT
        this.vocab_size = header.get(3); //V
        this.padded_vocab_size = header.get(7); // Vp
        this.num_layers = header.get(4); //L
        this.num_heads = header.get(5);  //NH
        this.channels = header.get(6);   //C
    }
}
