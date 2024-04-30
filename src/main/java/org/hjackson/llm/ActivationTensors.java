package org.hjackson.llm;

import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ActivationTensors {
  private static final Logger log = LoggerFactory.getLogger(ActivationTensors.class);
  public final static int NUM_ACTIVATION_TENSORS = 23;
  private final float[] mem;
  private final int encoded;
  private final float[] encoded_a;
  private final int encoded_size; // (B, T, C)
  private final float[] ln1_a;
  private final int ln1_size;
  private final int ln1;
  private final float[] ln1_mean_a;
  private final int ln1_mean_size;
  private final int ln1_mean;
  private float[] ln1_rstd_a;
  private int ln1_rstd_size;
  private int ln1_rstd;
  private final float[] qkv_a;
  private final int qkv_size;
  private final int qkv;
  private final float[] atty_a;
  private final int atty_size;
  private final int atty;
  private final float[] preatt_a;
  private final int preatt_size;
  private final int preatt;
  private final float[] att_a;
  private final int att_size;
  private final int att;
  private final float[] attproj_a;
  private final int attproj_size;
  private final int attproj;
  private final float[] residual2_a;
  private final int residual2_size;
  private final int residual2;
  private final float[] ln2_a;
  private final int ln2_size;
  private final int ln2;
  private float[] ln2_mean_a;
  private final int ln2_mean_size;
  private final int ln2_mean;
  private final float[] ln2_rstd_a;
  private final int ln2_rstd_size;
  private final int ln2_rstd;
  private float[] fch_a;
  private final int fch_size;
  private final int fch;
  private float[] fch_gelu_a;
  private final int fch_gelu_size;
  private final int fch_gelu;
  private final float[] fcproj_a;
  private final int fcproj_size;
  private final int fcproj;
  private float[] residual3_a;
  private final int residual3_size;
  private final int residual3;
  private float[] lnf_a;
  private final int lnf_size;
  private final int lnf;
  private float[] lnf_mean_a;
  private final int lnf_mean_size;
  private final int lnf_mean;
  private float[] lnf_rstd_a;
  private final int lnf_rstd_size;
  private final int lnf_rstd;
  private float[] logits_a;
  private final int logits_size;
  private final int logits;
  private float[] probs_a;
  private final int probs_size;
  private final int probs;
  private float[] losses_a;
  private int losses_size;

  private final int losses;

  private final int num_activations;

  public ActivationTensors(GPT2Config config, int B, int T) {

    int V = config.vocab_size;
    int L = config.num_layers;
    int NH = config.num_heads;
    int C = config.channels;

    encoded_size = B * T * C; // encoded
    encoded_a = new float[encoded_size];
    encoded = 0;

    ln1_size = L * B * T * C; // ln1
    ln1_a = new float[ln1_size];
    ln1 = encoded + encoded_size;

    ln1_mean_size = L * B * T;  // ln1_mean
    ln1_mean_a = new float[ln1_mean_size];
    ln1_mean = ln1 + ln1_size;

    ln1_rstd_size = L * B * T;  // ln1_rstd
    ln1_rstd_a = new float[ln1_rstd_size];
    ln1_rstd = ln1_mean + ln1_mean_size;

    qkv_size = L * B * T * 3 * C; // qkv
    qkv_a = new float[qkv_size];
    qkv = ln1_rstd + ln1_rstd_size;

    atty_size = L * B * T * C;  // atty
    atty_a = new float[atty_size];
    atty = qkv + qkv_size;

    preatt_size = L * B * NH * T * T;  // preatt
    preatt_a = new float[preatt_size];
    preatt = atty + atty_size;

    att_size = L * B * NH * T * T;  // att
    att_a = new float[att_size];
    att = preatt + preatt_size;

    attproj_size = L * B * T * C; // attproj
    attproj_a = new float[attproj_size];
    attproj = att + att_size;

    residual2_size = L * B * T * C; // residual2
    residual2_a = new float[residual2_size];
    residual2 = attproj + attproj_size;

    ln2_size = L * B * T * C; // ln2
    ln2_a = new float[ln2_size];
    ln2 = residual2 + residual2_size;

    ln2_mean_size = L * B * T; // ln2_mean
    ln2_mean_a = new float[ln2_mean_size];
    ln2_mean = ln2 + ln2_size;

    ln2_rstd_size = L * B * T; // ln2_rstd
    ln2_rstd_a = new float[ln2_rstd_size];
    ln2_rstd = ln2_mean + ln2_mean_size;

    fch_size = L * B * T * 4 * C; // fch
    fch_a = new float[fch_size];
    fch = ln2_rstd + ln2_rstd_size;

    fch_gelu_size = L * B * T * 4 * C; // fch_gelu
    fch_gelu_a = new float[fch_gelu_size];
    fch_gelu = fch + fch_size;

    fcproj_size = L * B * T * C; // fcproj
    fcproj_a = new float[fcproj_size];
    fcproj = fch_gelu + fch_gelu_size;

    residual3_size = L * B * T * C; // residual3
    residual3_a = new float[residual3_size];
    residual3 = fcproj + fcproj_size;

    lnf_size = B * T * C; // lnf
    lnf_a = new float[lnf_size];
    lnf = residual3 + residual3_size;

    lnf_mean_size = B * T; // lnf_mean
    lnf_mean_a = new float[lnf_mean_size];
    lnf_mean = lnf + lnf_size;

    lnf_rstd_size = B * T; // lnf_rstd
    lnf_rstd_a = new float[lnf_rstd_size];
    lnf_rstd = lnf_mean + lnf_mean_size;

    logits_size = B * T * V; // logits
    logits_a = new float[logits_size];
    logits = lnf_rstd + lnf_rstd_size;

    probs_size = B * T * V; // probs
    probs_a = new float[probs_size];
    probs = logits + logits_size;

    losses_size = B * T; // losses
    losses_a = new float[losses_size];
    losses = probs + probs_size;

    num_activations = losses + losses_size + 1;
    mem = new float[num_activations];
    Assert.floatEquals(getMem(residual3), 0.0f);
  }

  public int getNumActivations() {
    return num_activations;
  }

  public void zeroFill() {
    Arrays.fill(mem, 0.0f);
  }

  public void setLosses(int i, float dlossMean) {
    mem[losses + i] = dlossMean;
  }

  public int getLogits() {
    return logits;
  }

  public int getLosses() {
    return losses;
  }
  public int getProbs() {
    return probs;
  }

  public int getLnf() {
    return lnf;
  }

  public int getResidual3() {
    return residual3;
  }

  public int getLnfMean() {
    return lnf_mean;
  }

  public int getLnfRstd() {
    return lnf_rstd;
  }

  public int getEncoded() {
    return encoded;
  }

  public int getLn1() {
    return ln1;
  }

  public int getLn1Mean() {
    return ln1_mean;
  }

  public int getLn1Rstd() {
    return ln1_rstd;
  }

  public int getQkv() {
    return qkv;
  }

  public int getAtty() {
    return atty;
  }

  public int getAtt() {
    return att;
  }

  public int getResidual2() {
    return residual2;
  }

  public int getLn2() {
    return ln2;
  }

  public int getLn2Mean() {
    return ln2_mean;
  }

  public int getLn2Rstd() {
    return ln2_rstd;
  }

  public int getFch() {
    return fch;
  }

  public int getFchGelu() {
    return fch_gelu;
  }

  public int getPreatt() {
    return preatt;
  }

  public int getAttproj() {
    return attproj;
  }

  public int getFcproj() {
    return fcproj;
  }

  public float getMem(int i) {
    return mem[i];
  }

  public void setMem(Integer i, Float v) {
    mem[i] = v;
  }
}
