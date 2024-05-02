package org.hjackson.llm;

import java.io.IOException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class DataLoaderTest {
  private int B = 4;
  private int T = 64;
  private DataLoader dataLoader;
  @BeforeEach
  void setUp() throws Exception {
    int size = B * T + 1;
    int vals = size;
    int[] data = new int[size];
    for(int i = 0; i < size; i++) {
      data[i] = vals;
      vals--;
    }
    //dataLoader = new DataLoader("data/tiny_shakespeare_train.bin", B, T);
    dataLoader = new DataLoader(data, B, T);
  }

  @Test
  void testTargets() throws IOException {
    Assertions.assertTrue(dataLoader.targetsPresent());
    //dataLoader.dataloader_next_batch();
    Assertions.assertEquals(246, dataLoader.getTargets(10));
    Assertions.assertEquals(245, dataLoader.getTargets(11));
  }

}
