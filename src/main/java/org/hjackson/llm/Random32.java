package org.hjackson.llm;

import java.math.BigInteger;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

public class Random32 {
    public static Long RNG_STATE = Long.parseUnsignedLong("1337");

    private static long random_u32(Long state) {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        state ^= state >>> 12;
        state ^= state << 25;
        state ^= state >>> 27;
        RNG_STATE = state;

        BigInteger bi = BigInteger.valueOf(state);
        final BigInteger mul = bi.multiply(BigInteger.valueOf(0x2545F4914F6CDD1DL));
        long[] foo = new long[1];
        LongStream.range(32, 64).forEach((i) -> {
            if(mul.testBit((int) i)) {
                foo[0] |= (1L << (i-32));
                //System.out.printf("i == %d foo == %d   %d\n", i, foo[0], foo[0]*2);
            }
        });
        //System.out.printf("foo = %d mul == %s bitlength == %d state == %d\n", foo[0], mul + "", mul.bitLength(), state);
        Long tmp = foo[0];
        //System.out.printf("tmp == %d\n",tmp);
        return tmp;
    }

    public static float random_f32_(Long state) { // random float32 in [0,1)
         return random_u32(state) / 16777216.0f;
    }

    public static float random_f32(Long state) { // random float32 in [0,1)
        return (random_u32(state) >> 8) / 16777216.0f;
    }
}
