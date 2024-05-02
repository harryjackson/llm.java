package org.hjackson.llm;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.*;

public class ParameterTensors {
    public final int wte;
    private final int wte_size;
    private final int wpe;
    private final int wpe_size;
    private final int ln1w;
    private final int ln1w_size;
    private final int ln1b;
    private final int ln1b_size;
    private final int qkvw;
    private final int qkvw_size;
    private final int qkvb;
    private final int qkvb_size;
    private final int attprojw;
    private final int attprojw_size;
    private final int attprojb;
    private final int attprojb_size;
    private final int ln2w;
    private final int ln2w_size;
    private final int ln2b;
    private final int ln2b_size;
    private final int fcw;
    private final int fcw_size;
    private final int fcb;
    private final int fcb_size;
    private final int fcprojw;
    private final int fcprojw_size;
    private final int fcprojb;
    private final int fcprojb_size;
    private final int lnfw;
    private final int lnfw_size;
    private final int lnfb;
    private final int lnfb_size;
    private final int num_params;
    public final float[] mem;
    private boolean ok = false;
    private final Map<Integer, Float> tracking = new HashMap<>();
    public ParameterTensors(MemorySegment segment, GPT2Config config) {
        this(config);
        int pos = 1024;//header
        for (int i = 0; i < num_params; i++, pos += 4) {
            mem[i] = segment.get(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.nativeOrder()), pos);
        }
        Assert.floatEquals(this.mem[0], -0.11010301f);
        runParamAssertions();
        ok = true;
    }
    public ParameterTensors(GPT2Config config) {
        int maxT = config.max_seq_len;
        int C = config.channels;
        int V = config.vocab_size;
        int Vp = config.padded_vocab_size;
        int L = config.num_layers;
        // allocate space for all the parameters and read them in
        wte_size = Vp * C;
        wte = 0;
        wpe_size = maxT * C;
        wpe = wte + wte_size;
        ln1w_size = L * C;
        ln1w = wpe + wpe_size;
        ln1b_size = L * C;
        ln1b = ln1w + ln1w_size;
        qkvw_size = L * (3 * C) * C;
        qkvw = ln1b + ln1b_size;
        qkvb_size = L * (3 * C);
        qkvb = qkvw + qkvw_size;
        attprojw_size = L * C * C;
        attprojw = qkvb + qkvb_size;
        attprojb_size = L * C;
        attprojb = attprojw + attprojw_size;
        ln2w_size = L * C;
        ln2w = attprojb + attprojb_size;
        ln2b_size = L * C;
        ln2b = ln2w + ln2w_size;
        fcw_size = L * (4 * C) * C;
        fcw = ln2b + ln2b_size;
        fcb_size = L * (4 * C);
        fcb = fcw + fcw_size;
        fcprojw_size = L * C * (4 * C);
        fcprojw = fcb + fcb_size;
        fcprojb_size = L * C;
        fcprojb = fcprojw + fcprojw_size;
        lnfw_size = C;
        lnfw = fcprojb + fcprojb_size;
        lnfb_size = C;
        lnfb = lnfw + lnfw_size;
        num_params = lnfb + lnfb_size;
        mem = new float[num_params];
        tracking.put(58904066, 0.0f);
    }
    public boolean ok() {
        return ok;
    }
    public boolean didChange(String s) {//debugging mem access
        boolean res = false;
        for(Integer k : tracking.keySet()) {
            float curr = mem[k];
            float prev = tracking.get(k);
            if(Float.compare(curr, prev) != 0) {
                res = true;
                System.out.printf("tracking change %d %f -> %f\n", k, prev, curr);
                tracking.put(k, curr);
            }
        }
        return res;
    }
    private void runParamAssertions() {
        //I'm not looping because I like to know what line failed in the stack trace
        Assert.floatEquals(mem[wte], -0.11010301113128662f);
        Assert.floatEquals(mem[wte + 5], -0.078917674720287323f);
        Assert.floatEquals(mem[wpe], -0.018820719793438911f);
        Assert.floatEquals(mem[wpe + 1], -0.197418600320816040f);
        Assert.floatEquals(mem[wpe + 5], -0.105013281106948853f);
        Assert.floatEquals(mem[ln1w + 1], 0.181958660483360291f);
        Assert.floatEquals(mem[ln1w + 5], 0.194811657071113586f);
        Assert.floatEquals(mem[ln1b], -0.003677325090393424f);
        Assert.floatEquals(mem[ln1b + 5], -0.011468173004686832f);
        Assert.floatEquals(mem[qkvw], -0.473848402500152588f);
        Assert.floatEquals(mem[qkvw + 5], 0.032973293215036392f);
        Assert.floatEquals(mem[qkvb], 0.480339139699935913f);
        Assert.floatEquals(mem[qkvb + 5], -0.095427356660366058f);
        Assert.floatEquals(mem[attprojw], 0.312718182802200317f);
        Assert.floatEquals(mem[attprojw + 5], -0.437642186880111694f);
        Assert.floatEquals(mem[attprojb], 0.150291591882705688f);
        Assert.floatEquals(mem[attprojb + 5], -0.034447547048330307f);
        Assert.floatEquals(mem[ln2w], 0.130966052412986755f);
        Assert.floatEquals(mem[ln2w + 5], 1.269531369209289551f);
        Assert.floatEquals(mem[ln2b], 0.042478270828723907f);
        Assert.floatEquals(mem[ln2b + 5], -0.026806578040122986f);
        Assert.floatEquals(mem[fcw], 0.094201952219009399f);
        Assert.floatEquals(mem[fcw + 5], 0.051278203725814819f);
        Assert.floatEquals(mem[fcb], 0.039619479328393936f);
        Assert.floatEquals(mem[fcb + 5], -0.014704782515764236f);
        Assert.floatEquals(mem[fcprojw], -0.106606408953666687f);
        Assert.floatEquals(mem[fcprojw + 5], -0.105633556842803955f);
        Assert.floatEquals(mem[fcprojb], 0.045023269951343536f);
        Assert.floatEquals(mem[fcprojb + 5], -0.238876312971115112f);
        Assert.floatEquals(mem[lnfw], 1.397080421447753906f);
        Assert.floatEquals(mem[lnfw + 5], 1.250811934471130371f);
        Assert.floatEquals(mem[lnfb], 0.001087164739146829f);
        Assert.floatEquals(mem[lnfb + 5], -0.071351118385791779f);
    }
    public int getWte() {
        return wte;
    }
    public int getNumParams() {
        return num_params;
    }
    public void zeroFill() { Arrays.fill(mem, 0); }
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
    public int getFcb() { return fcb;}
    public int getFcprojb() {
        return fcprojb;
    }
    public int getWpe() {
        return wpe;
    }
}