import logging
import array
import struct
import numpy as np

log = logging.getLogger(__name__)


class Config:
    headers = 256
    header_size_b = headers * 4
    vocab_size = 0
    num_layers = 0
    num_heads = 0
    channels = 0


class ParameterTensors:
    def __init__(self, file_name, config):

        f = open(file_name, 'rb')
        f.seek(1024,0)

        self.file_size = f.tell()
        assert((self.file_size - 1024) % 4 == 0)
        self.num_params = int((self.file_size - 1024)/4)
        self.mem = np.fromfile(f, np.float32)
        struct.unpack('f' * self.num_params, f.read(4 * self.num_params))
        self.maxT = config.max_seq_len
        self.C = config.channels
        self.V = config.vocab_size
        self.L = config.num_layers

        self.wte_size = self.V * self.C
        self.wte = 0

        self.wpe_size = self.maxT * self.C
        self.wpe = self.wte + self.wte_size

        self.ln1w_size = self.L * self.C
        self.ln1w = self.wpe + self.wpe_size

        self.ln1b_size = self.L * self.C
        self.ln1b = self.ln1w + self.ln1w_size

        self.qkvw_size = self.L * (3 * self.C) * self.C
        self.qkvw = self.ln1b + self.ln1b_size

        self.qkvb_size = self.L * (3 * self.C)
        self.qkvb = self.qkvw + self.qkvw_size

        self.attprojw_size = self.L * self.C * self.C
        self.attprojw = self.qkvb + self.qkvb_size

        self.attprojb_size = self.L * self.C
        self.attprojb = self.attprojw + self.attprojw_size

        self.ln2w_size = self.L * self.C
        self.ln2w = self.attprojb + self.attprojb_size

        self.ln2b_size = self.L * self.C
        self.ln2b = self.ln2w + self.ln2w_size

        self.fcw_size = self.L * (4 * self.C) * self.C
        self.fcw = self.ln2b + self.ln2b_size

        self.fcb_size = self.L * (4 * self.C)
        self.fcb = self.fcw + self.fcw_size

        self.fcprojw_size = self.L * self.C * (4 * self.C)
        self.fcprojw = self.fcb + self.fcb_size

        self.fcprojb_size = self.L * self.C
        self.fcprojb = self.fcprojw + self.fcprojw_size

        self.lnfw_size = self.C
        self.lnfw = self.fcprojb + self.fcprojb_size

        self.lnfb_size = self.C
        self.lnfb = self.lnfw + self.lnfw_size

        self.num_params = self.lnfb + self.lnfb_size
        log.info(f"num_params {self.num_params}")
        self.run_assertions()

    def getMem(self, ix):
        return self.mem[ix]

    def run_assertions(self):
        v = self.mem[self.wte]
        print(f'wte[0] == {v}')
        assert(self.mem[self.wte] == -0.11010301113128662)
        assert(self.mem[self.wte + 5] == -0.078917674720287323)

        assert (self.mem[self.wpe] == -0.018820719793438911)
        assert (self.mem[self.wpe + 1] == -0.197418600320816040)
        assert (self.getMem(self.wpe + 5) == -0.105013281106948853)

        assert (self.getMem(self.ln1w + 1) == 0.181958660483360291)

        assert (self.getMem(self.ln1w + 5) == 0.194811657071113586)

        assert (self.getMem(self.ln1b) == -0.003677325090393424)
        assert (self.getMem(self.ln1b + 5) == -0.011468173004686832)

        assert (self.getMem(self.qkvw) == -0.473848402500152588)
        assert (self.getMem(self.qkvw + 5) == 0.032973293215036392)

        assert (self.getMem(self.qkvb) == 0.480339139699935913)
        assert (self.getMem(self.qkvb + 5) == -0.095427356660366058)

        assert (self.getMem(self.attprojw) == 0.312718182802200317)
        assert (self.getMem(self.attprojw + 5) == -0.437642186880111694)

        assert (self.getMem(self.attprojb) == 0.150291591882705688)
        assert (self.getMem(self.attprojb + 5) == -0.034447547048330307)

        assert (self.getMem(self.ln2w) == 0.130966052412986755)
        assert (self.getMem(self.ln2w + 5) == 1.269531369209289551)

        assert (self.getMem(self.ln2b) == 0.042478270828723907)
        assert (self.getMem(self.ln2b + 5) == -0.026806578040122986)

        assert (self.getMem(self.fcw) == 0.094201952219009399)
        assert (self.getMem(self.fcw + 5) == 0.051278203725814819)

        assert (self.getMem(self.fcb) == 0.039619479328393936)
        assert (self.getMem(self.fcb + 5) == -0.014704782515764236)

        assert (self.getMem(self.fcprojw) == -0.106606408953666687)
        assert (self.getMem(self.fcprojw + 5) == -0.105633556842803955)

        assert (self.getMem(self.fcprojb) == 0.045023269951343536)
        assert (self.getMem(self.fcprojb + 5) == -0.238876312971115112)

        assert (self.getMem(self.lnfw) == 1.397080421447753906)
        assert (self.getMem(self.lnfw + 5) == 1.250811934471130371)

        print(f'printing {self.getMem(self.lnfb)}\n')
        assert (self.getMem(self.lnfb) == 0.00108716473914682)
        assert (self.getMem(self.lnfb + 5) == -0.071351118385791779)





