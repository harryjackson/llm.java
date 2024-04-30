package org.hjackson.llm;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;

public class Tokenizer {
    private static final Logger log = LoggerFactory.getLogger(Tokenizer.class);
    private final int vocab_size;
    private final String[] token_table;
    public final int init_ok;
    private RandomAccessFile file;
    private String fileName;
    private final int[] header = new int[256];
    //private final int init_ok;

    public Tokenizer(String filename) throws IOException {
        fileName = filename;

        File f = new File(filename);

        if (!f.exists()) {
            // try to be more helpful as we just added this feature, erase later
            log.error(
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
        log.info("header[0] == {}", header[0]);
        log.info("header[0] == {}", header[2]);
        assert (header[0] == 20240328);
        assert (header[1] == 1);
        vocab_size = header[2];
        log.info("Vocab Size == {}", vocab_size);
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
            //log.info("length ==  {}", length);
            assert(length > 0);
            byte[] tmp = new byte[length];
            int t = file.read(tmp, 0, length);
            String tok = new String(tmp, StandardCharsets.UTF_8);
            //log.debug("read {} bytes char == >{}<", length, tok);
            //assert(length > 0); // every token should be at least one character
//            char *token_bytes = (char *)malloc(length + 1);
//            fread(token_bytes, sizeof(char), length, file);
//            token_bytes[length] = '\0';  // Add null terminator for printing

            token_table[n++] = tok;
            pos = pos + 1 + length;
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
            log.error("invalid token id {}!\n", token_id);
            throw new IllegalStateException(STR."Something bad happened, bad token_id: \{token_id}");
        }
    }


}

