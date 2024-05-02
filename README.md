# llm.java

Large Language Model (LLM) example in java i.e. GPT2. This is a port of 
the [Llm.c code that lives here](https://github.com/karpathy/llm.c) written 
by @[karpathy](https://github.com/karpathy) 

## Before Running ChatGPT2 in Java

Before attempting to run this some prep work needs to happen. If you check 
the [llm.c repository](https://github.com/karpathy/llm.c) these steps are very similar. 
The reason the same code is in this repository is because LLM.c is still a moving target.

I highly recommend running the original llm.c to see it work. It's wonderful.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepro_tinyshakespeare.py
python train_gpt2.py
```

### JVM Requirements

I used the GraalVM for this running version 21. If you're using sdkman.

```bash
sdk default java 21.0.2-graalce
```

I tested the following JVM version and they all seem to work. I have not investigated why some are slower than 
others.

1. Temurin: This ran at half the speed of Graal. I stopped it at step 10

```bash
sdk install java 21-tem
sdk use java 21-tem
```

2. Correto: This VM was also really slow compared to Graal. So I stopped it after step 10

```bash
sdk install java 21.0.3-amzn
sdk use java 21.0.3-amzn
```


## Running

Note the arguments passed to the JVM. Of particular note is "-Djava.util.concurrent.ForkJoinPool.common.parallelism=10", 
adjust this based on how many cores you have. The matrix multiplication methods are entirely CPU bound so adding more
threads than cores will just slow things down. 

```bash
mvn clean install;
java -jar -ea --add-modules jdk.incubator.vector --enable-preview -Xmx8g -Djava.util.concurrent.ForkJoinPool.common.parallelism=10 target/gpt2-1.0-SNAPSHOT.jar
```

## Performance

I've made no attempt to tune this for performance. The C version is still much faster than this version. There are 
some low-hanging fruit like parallelizing some of the loops. I made the matmul_forward and matmul_backward both 
parallel because it was painfully slow without it.
