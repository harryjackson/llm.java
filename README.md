# llm.java

Large Language Model (LLM) example in java i.e. GPT2

## Before Running ChatGPT2 in Java

Before attempting to run this some prep work needs to happen. If you check 
the [llm.c repository](https://github.com/karpathy/llm.c) these steps are similar. 
The reason the same code is in this repository is because LLM.c is still a moving target
and I had some breaking changes with version 3 of gpt2_124M.bin.

I highly recommend running llm.c to see it work.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepro_tinyshakespeare.py
python train_gpt2.py
make train_gpt2fp32cu
```

### JVM Requirements

I used the GraalVM for this running version 21. If you're using sdkman.

```bash
sdk default java 21.0.2-graalce
```

## Running

Note the arguments passed to the JVM.

```bash
mvn clean install;
java -jar -ea --add-modules jdk.incubator.vector --enable-preview -Xmx8g -Djava.util.concurrent.ForkJoinPool.common.parallelism=10
```

## Performance

I have made no attempt to tune this for performance. The C version is still much faster than this version. There are 
some low hanging fruit like parallelizing some of the loops. I made the matmul_forward and backward both parallel 
because it was painfully slow without it.
