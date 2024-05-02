import numpy as np
import logging
import sys
import struct


def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))


def main():
    #logging.basicConfig(filename='myapp.log', level=logging.INFO)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    filename = '/git/llm/gpt2_124M.bin'

    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    print(binary(data[int(sys.argv[1])]))

if __name__ == '__main__':
    main()