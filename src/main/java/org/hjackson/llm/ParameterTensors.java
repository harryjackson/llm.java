package org.hjackson.llm;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ParameterTensors {
    private static final Logger log = LoggerFactory.getLogger(ParameterTensors.class);
    public final static int NUM_PARAMETER_TENSORS = 16;
    public final int wte;
    public final float[] wte_a;
    private final int wte_size;
    private final int wpe;
    private final float[] wpe_a;
    private final int wpe_size;
    private final int ln1w;
    private final float[] ln1w_a;
    private final int ln1w_size;
    private final int ln1b;
    private final float[] ln1b_a;
    private final int ln1b_size;
    private final int qkvw;
    private final float[] qkvw_a;
    private final int qkvw_size;
    private final int qkvb;
    private final float[] qkvb_a;
    private final int qkvb_size;
    private final int attprojw;
    private final float[] attprojw_a;
    private final int attprojw_size;
    private final int attprojb;
    private final float[] attprojb_a;
    private final int attprojb_size;
    private final int ln2w;
    private final float[] ln2w_a;
    private final int ln2w_size;
    private final int ln2b;
    private final float[] ln2b_a;
    private final int ln2b_size;
    private final int fcw;
    private final float[] fcw_a;
    private final int fcw_size;
    private final int fcb;
    private final float[] fcb_a;
    private final int fcb_size;
    private final int fcprojw;
    private final float[] fcprojw_a;
    private final int fcprojw_size;
    private final int fcprojb;
    private final float[] fcprojb_a;
    private final int fcprojb_size;
    private final int lnfw;
    private final float[] lnfw_a;
    private final int lnfw_size;
    private final int lnfb;
    private final float[] lnfb_a;
    private final int lnfb_size;
    private final int num_params;

    private final float[] mem;
    private boolean ok = false;

    public ParameterTensors(MemorySegment segment, GPT2Config config) {
        this(config);
        int pos = 1024;
        int test = 497760256/4 - 256;
        //log.info("num_params == {} test == {}", num_params, test);
        Assert.intEquals(num_params, test);
        for (int i = 0; i < num_params; i++, pos += 4) {
            mem[i] = (float) segment.get(ValueLayout.JAVA_FLOAT, pos);
            //mem[i] = segment.getAtIndex(ValueLayout.JAVA_float, i);
        }

//        pos = 1024;
//        for (int i = 0; i < wte_size; i++, pos += 4) {
//            wte_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < wpe_size; i++, pos += 4) {
//            wpe_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < ln1w_size; i++, pos += 4) {
//            ln1w_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < ln1b_size; i++, pos += 4) {
//            ln1b_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < qkvw_size; i++, pos += 4) {
//            qkvw_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < qkvb_size; i++, pos += 4) {
//            qkvb_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < attprojw_size; i++, pos += 4) {
//            attprojw_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < attprojb_size ; i++, pos += 4) {
//            attprojb_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < ln2w_size; i++, pos += 4) {
//            ln2w_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < ln2b_size; i++, pos += 4) {
//            ln2b_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < fcw_size; i++, pos += 4) {
//            fcw_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < fcb_size; i++, pos += 4) {
//            fcb_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < fcprojw_size; i++, pos += 4) {
//            fcprojw_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < fcprojb_size; i++, pos += 4) {
//            fcprojb_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < lnfw_size; i++, pos += 4) {
//            lnfw_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
//        for (int i = 0; i < lnfb_size; i++, pos += 4) {
//            //log.info("i=={} pos=={}", i, pos);
//            lnfb_a[i] = segment.get(ValueLayout.JAVA_float, pos);
//        }
        //log.info("{}", this.getMem(0));
        Assert.floatEquals(this.getMem(0), -0.11010301f);
        runParamAssertions();
        ok = true;
    }

    public ParameterTensors(GPT2Config config) {
        int maxT = config.max_seq_len;
        int C = config.channels;
        int V = config.vocab_size;
        int L = config.num_layers;
        assert(C > 0);

        // allocate space for all the parameters and read them in
        wte_size = V * C;
        wte = 0;
        wte_a = new float[wte_size];

        wpe_size = maxT * C;
        wpe = wte + wte_size;
        wpe_a = new float[wte_size];

        ln1w_size = L * C;
        ln1w = wpe + wpe_size;
        ln1w_a = new float[ln1w_size];

        ln1b_size = L * C;
        ln1b = ln1w + ln1w_size;
        ln1b_a = new float[ln1b_size];

        qkvw_size = L * (3 * C) * C;
        qkvw = ln1b + ln1b_size;
        qkvw_a = new float[qkvw_size];

        qkvb_size = L * (3 * C);
        qkvb = qkvw + qkvw_size;
        qkvb_a = new float[qkvb_size];

        attprojw_size = L * C * C;
        attprojw = qkvb + qkvb_size;
        attprojw_a = new float[attprojw_size];

        attprojb_size = L * C;
        attprojb = attprojw + attprojw_size;
        attprojb_a = new float[attprojb_size];

        ln2w_size = L * C;
        ln2w = attprojb + attprojb_size;
        ln2w_a = new float[ln2w_size];

        ln2b_size = L * C;
        ln2b = ln2w + ln2w_size;
        ln2b_a = new float[ln2b_size];

        fcw_size = L * (4 * C) * C;
        fcw = ln2b + ln2b_size;
        fcw_a = new float[fcw_size];

        fcb_size = L * (4 * C);
        fcb = fcw + fcw_size;
        fcb_a = new float[fcb_size];

        fcprojw_size = L * C * (4 * C);
        fcprojw = fcb + fcb_size;
        fcprojw_a = new float[fcprojw_size];

        fcprojb_size = L * C;
        fcprojb = fcprojw + fcprojw_size;
        fcprojb_a = new float[fcprojb_size];

        lnfw_size = C;
        lnfw = fcprojb + fcprojb_size;
        lnfw_a = new float[lnfw_size];

        lnfb_size = C;
        lnfb = lnfw + lnfw_size;
        lnfb_a = new float[lnfb_size];

        num_params = lnfb + lnfb_size;
        log.info("num_params {}", num_params);
        assert (num_params > 100000);

        mem = new float[num_params];
        //log.info("num params == {}", num_params);
    }

    public boolean ok() {
        return ok;
    }
    private void runParamAssertions() {

        Assert.floatEquals(mem[wte], -0.11010301113128662f);
        Assert.floatEquals(mem[wte + 5], -0.078917674720287323f);

        Assert.floatEquals(mem[wpe], -0.018820719793438911f);
        Assert.floatEquals(mem[wpe + 1], -0.197418600320816040f);
        Assert.floatEquals(this.getMem(wpe + 5), -0.105013281106948853f);

        Assert.floatEquals(this.getMem(ln1w + 1), 0.181958660483360291f);

        Assert.floatEquals(this.getMem(ln1w + 5), 0.194811657071113586f);

        Assert.floatEquals(this.getMem(ln1b), -0.003677325090393424f);
        Assert.floatEquals(this.getMem(ln1b + 5), -0.011468173004686832f);

        Assert.floatEquals(this.getMem(qkvw), -0.473848402500152588f);
        Assert.floatEquals(this.getMem(qkvw + 5), 0.032973293215036392f);

        Assert.floatEquals(this.getMem(qkvb), 0.480339139699935913f);
        Assert.floatEquals(this.getMem(qkvb + 5), -0.095427356660366058f);

        Assert.floatEquals(this.getMem(attprojw), 0.312718182802200317f);
        Assert.floatEquals(this.getMem(attprojw + 5), -0.437642186880111694f);

        Assert.floatEquals(this.getMem(attprojb), 0.150291591882705688f);
        Assert.floatEquals(this.getMem(attprojb + 5), -0.034447547048330307f);

        Assert.floatEquals(this.getMem(ln2w), 0.130966052412986755f);
        Assert.floatEquals(this.getMem(ln2w + 5), 1.269531369209289551f);

        Assert.floatEquals(this.getMem(ln2b), 0.042478270828723907f);
        Assert.floatEquals(this.getMem(ln2b + 5), -0.026806578040122986f);

        Assert.floatEquals(this.getMem(fcw), 0.094201952219009399f);
        Assert.floatEquals(this.getMem(fcw + 5), 0.051278203725814819f);

        Assert.floatEquals(this.getMem(fcb), 0.039619479328393936f);
        Assert.floatEquals(this.getMem(fcb + 5), -0.014704782515764236f);

        Assert.floatEquals(this.getMem(fcprojw), -0.106606408953666687f);
        Assert.floatEquals(this.getMem(fcprojw + 5), -0.105633556842803955f);

        Assert.floatEquals(this.getMem(fcprojb), 0.045023269951343536f);
        Assert.floatEquals(this.getMem(fcprojb + 5), -0.238876312971115112f);

        Assert.floatEquals(this.getMem(lnfw), 1.397080421447753906f);
        Assert.floatEquals(this.getMem(lnfw + 5), 1.250811934471130371f);

        Assert.floatEquals(this.getMem(lnfb), 0.001087164739146829f);
        Assert.floatEquals(this.getMem(lnfb + 5), -0.071351118385791779f);

    }

    public int getWte() {
        return wte;
    }

    public int getNumParams() {
        return num_params;
    }

    public void zeroFill() {
      Arrays.fill(mem, 0);
    }

    public int getLnfw() {
        return lnfw;
    }

    public int getLnfb() {
        return lnfb;
    }

    public int getQkvw() {
        return qkvw;
    }

    public int getAttprojw() {
        return attprojw;
    }

    public int getLn2w() {
        return ln2w;
    }

    public int getFcw() {
        return fcw;
    }

    public int getFcprojw() {
        return fcprojw;
    }

    public int getLn1w() {
        return ln1w;
    }

    public int getLn1b() {
        return ln1b;
    }

    public int getQkvb() {
        return qkvb;
    }

    public int getAttprojb() {
        return attprojb;
    }

    public int getLn2b() {
        return ln2b;
    }

    public int getFcb() {
        return fcb;
    }

    public int getFcprojb() {
        return fcprojb;
    }

    public int getWpe() {
        return wpe;
    }

    public float getMem(int i) {
        return mem[i];
    }

    public void setMem(int i, float v) {
        mem[i] = v;
    }
}