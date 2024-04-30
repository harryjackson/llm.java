package org.hjackson.llm;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Assertions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Assert {
    private static final Logger log = LoggerFactory.getLogger(Assert.class);
    public static final float EPSILON = 1e-6f;
    private static final Map<String, Float> check = new HashMap<>();

    static {
//        check.put("l_ln1w134", 0.223320335f);
//        check.put("l_ln1w145", 0.223220333f);
//        check.put("l_ln1w45", 0.223220333f);
        check.put("l_ln1w1", 0.223220333f);
        check.put("qkv13", 0.49580425024032592f);
        check.put("qkv49", 0.48021019f);
        check.put("l_residual313", 0.391122013330459f);
        check.put("ln11", 0.01925184391438961f);
    }
    public static void floatEquals(float a, float b) {
        float abs = Math.abs(a - b);
        if(!nearlyEqual(a, b, EPSILON)) {
            throw new IllegalStateException("float diff too big " + abs);
        }
        if(abs > EPSILON) {
            throw new IllegalStateException("float diff too big " + abs);
        }
    }
    public static void intEquals(int a, int b) {
        Assertions.assertEquals(a, b);
    }
    public static void mod4(int t) {
        if(t % 4 != 0) {
            throw new IllegalStateException(STR."Found alignment Issue:\{t}");
        }
    }
    public static void nonNan(final float meanLoss) {
        if(Float.isNaN(meanLoss)) {
            throw new IllegalStateException("NaN Found");
        }
    }
    public static void atLayerCheck(final String label, final long layerCount, final float val) {
        String id = label + layerCount;
        //log.info("comp {} {}", id, val);
        Float f = check.get(id);
        if(f != null) {
            try {
                log.info("{} comp {} == {} diff == {} {}", id, f, val, Math.abs(val - f), EPSILON);
                floatEquals(f, val);
            }
            catch (IllegalStateException e) {
                //String msg = STR."Expected float \{f} == \{val} at label \{id}";
                String m = String.format("Expected float %18.18f == %18.18f at label %s", f, val, label);
                throw new IllegalStateException(m, e);
            }
         }
    }

    public static boolean nearlyEqual(float a, float b, float epsilon) {
        final float absA = Math.abs(a);
        final float absB = Math.abs(b);
        final float diff = Math.abs(a - b);

        if (a == b) { // shortcut, handles infinities
            return true;
        } else if (a == 0 || b == 0 || diff < Float.MIN_NORMAL) {
            // a or b is zero or both are extremely close to it
            // relative error is less meaningful here
            return diff < (epsilon * Float.MIN_NORMAL);
        } else { // use relative error
            return diff / (absA + absB) < epsilon;
        }
    }
}
