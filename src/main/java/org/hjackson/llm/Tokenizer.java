package org.hjackson.llm;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.rmi.UnexpectedException;

public class Tokenizer {
    private final int vocab_size;
    private final String[] token_table;
    public final int init_ok;
    final int end_of_text;
    private RandomAccessFile file;
    private String fileName;
    private final int[] header = new int[256];
    //private final int init_ok;

    public Tokenizer(String filename) throws IOException {
        fileName = filename;

        File f = new File(filename);

        if (!f.exists()) {
            // try to be more helpful as we just added this feature, erase later
            System.out.printf(
                    """
                                  ---
                                  WARNING: Failed to open the tokenizer file {}
                                           The Tokenizer is a new feature added April 14 2024.\n
                                           "Re-run `python train_gpt2.py` to write it\\n");
                                  ---       
                            """, filename);
        }
        file = new RandomAccessFile(f, "r");
        // read in the header
        for (int i = 0; i < 256; i++) {
            //file has integers in little endian, jvm is bigendian
            header[i] = Integer.reverseBytes(file.readInt());
        }
        System.out.printf("header[0] == %d\n", header[0]);
        System.out.printf("header[0] == %d\n", header[2]);
        assert (header[0] == 20240328);
        assert (header[1] == 2);
        vocab_size = header[2];
        end_of_text = header[3];
        if(end_of_text != 50256) {
            throw new UnexpectedException("Something has changed");
        }
        System.out.printf("Vocab Size == %d end_of_text == %d\n", vocab_size, end_of_text);
        // read in all the tokens
        int length;
        token_table = new String[vocab_size];
        int pos = 1024;
        int vocabEnd =  vocab_size + 1024;
        int n = 0;
        for (int i = pos; i < vocabEnd; i++) {
            file.seek(pos);
            byte l = file.readByte();
            length = l & 0xff;// java uses signed bytes, convert to unsigned
            assert(length > 0);
            byte[] tmp = new byte[length];
            int t = file.read(tmp, 0, length);
            String tok = new String(tmp, StandardCharsets.UTF_8);
            //System.out.printf(tok);
            token_table[n++] = tok;
            pos = pos + 1 + length;
            //System.out.printf(">%s<\n", tok);
        }
        file.close();
        init_ok = 1;
    }

    String tokenizer_decode(int token_id) {
        if (init_ok == 0) {
            return null;
        }
        if (token_id < vocab_size) {
            return token_table[token_id];
        } else {
            System.out.printf("invalid token id {}!\n", token_id);
            throw new IllegalStateException("Something bad happened, bad token_id: " + token_id);
        }
    }
}

