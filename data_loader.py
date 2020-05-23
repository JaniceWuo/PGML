# this code from https://github.com/nlpcl-lab/sentential_argument_generation/blob/master/data_loader.py
import os,struct
import json
from queue import Queue
from threading import Thread
from random import shuffle, sample, randint
from tensorflow.core.example import example_pb2
import numpy as np


def sample_generator(bin_fname, single_pass=False):
    """
    Generator that reads binary file and yield text sample for training or inference.
    setname(str): binary file that one of ['train', 'dev', 'test']
    single_pass(boolean): If True, iterate the whole dataset only once (for testcase)
    """
    assert os.path.exists(bin_fname)
    print("the bin_fname:",bin_fname)

    while True:  # If single_pass, escape this loop after one epoch!  
        reader = open(bin_fname, 'rb')
        while True:
            len_bytes = reader.read(8)   
            if not len_bytes:
                print("Reading one file is end!")    
                break  # Break if file is end  确实应该break
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            example = example_pb2.Example.FromString(example_str)

            if 'wikitext' not in bin_fname:
                enc_text = example.features.feature['enc'].bytes_list.value[0].decode()
                dec_text = example.features.feature['dec'].bytes_list.value[0].decode()
                cid = example.features.feature['cid'].bytes_list.value[0].decode()
                pid = example.features.feature['pid'].bytes_list.value[0].decode()
                ppid = example.features.feature['ppid'].bytes_list.value[0].decode()
                yield (enc_text, dec_text, cid, pid, ppid)

            else:
                text = example.features.feature['text'].bytes_list.value[0].decode()
                if len(text.split()) < 3: continue
                yield text

        if single_pass:
            print("Single pass is end!")
            break



class Example:
    def __init__(self, enc_text, dec_text, voca, hps):
        self.hps, self.vocab = hps, voca

        enc_toks, dec_toks = enc_text.split()[:self.hps.max_enc_len], dec_text.split()

        self.enc_len = len(enc_toks)
        self.enc_input = self.vocab.text2ids(enc_toks)
        self.dec_input = self.vocab.text2ids(dec_toks)

        self.original_enc_text = enc_text
        self.original_dec_text = dec_text

        def make_dec_inp_tgt_seq(seq, max_len, beg_id, eos_id):
            inp = [beg_id] + seq[:]
            tgt = seq[:]

            if len(seq) > max_len:
                inp = inp[:max_len]
                tgt = tgt[:max_len]
            else:
                tgt.append(eos_id)
            assert len(inp) == len(tgt)
            # print(inp,tgt)
            return inp, tgt

        self.dec_input, self.dec_target = make_dec_inp_tgt_seq(self.dec_input, self.hps.max_dec_len, self.vocab.beg_id,
                                                               self.vocab.eos_id)

        self.dec_len = len(self.dec_input)

    def pad_enc_input(self, max_len):
        while len(self.enc_input) < max_len:
            self.enc_input.append(self.vocab.pad_id)

    def pad_dec_inp_tgt(self, dec_max_len):
        assert len(self.dec_input) == len(self.dec_target)
        while len(self.dec_input) < dec_max_len:
            self.dec_input.append(self.vocab.pad_id)
        while len(self.dec_target) < dec_max_len:
            self.dec_target.append(self.vocab.pad_id)

class Example_:
    def __init__(self, enc_text, dec_text, cid, pid, ppid, voca, hps):
        self.hps, self.vocab = hps, voca
        self.cid, self.pid, self.ppid = cid, pid, ppid

        enc_toks, dec_toks = enc_text.split()[:self.hps.max_enc_len], dec_text.split()

        self.enc_len = len(enc_toks)
        self.enc_input = self.vocab.text2ids(enc_toks)
        self.dec_input = self.vocab.text2ids(dec_toks)

        self.original_enc_text = enc_text
        self.original_dec_text = dec_text

        def make_dec_inp_tgt_seq(seq, max_len, beg_id, eos_id):
            inp = [beg_id] + seq[:]
            tgt = seq[:]

            if len(seq) > max_len:
                inp = inp[:max_len]
                tgt = tgt[:max_len]
            else:
                tgt.append(eos_id)
            assert len(inp) == len(tgt)
            # print(inp,tgt)
            return inp, tgt

        self.dec_input, self.dec_target = make_dec_inp_tgt_seq(self.dec_input, self.hps.max_dec_len, self.vocab.beg_id,
                                                              self.vocab.eos_id)

        self.dec_len = len(self.dec_input)

    def pad_enc_input(self, max_len):
        while len(self.enc_input) < max_len:
            self.enc_input.append(self.vocab.pad_id)

    def pad_dec_inp_tgt(self, dec_max_len):
        assert len(self.dec_input) == len(self.dec_target)
        while len(self.dec_input) < dec_max_len:
            self.dec_input.append(self.vocab.pad_id)
        while len(self.dec_target) < dec_max_len:
            self.dec_target.append(self.vocab.pad_id)



class Batch():
    def __init__(self, example_list, hps, vocab,b_size):   
        self.hps = hps
        self.vocab = vocab
        self.bsize = b_size
        self.init_enc_seq(example_list)
        self.init_dec_seq(example_list)
        self.save_original_seq(example_list)
        

    def init_enc_seq(self, example_list):
        max_enc_len = max([ex.enc_len for ex in example_list])  #
        for ex in example_list:
            ex.pad_enc_input(max_enc_len)
        # print("batch_size:",self.batch_size)
        self.enc_batch = np.zeros((self.bsize, max_enc_len), dtype=np.int32)
        
        self.enc_lens = np.zeros((self.bsize), dtype=np.int32)
        self.enc_pad_mask = np.zeros((self.bsize, max_enc_len), dtype=np.float32)

        # Fill enc batch
        for idx, ex in enumerate(example_list):
            # print("enc_input",ex.enc_input[:])
            self.enc_batch[idx, :] = ex.enc_input[:]
            self.enc_lens[idx] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_pad_mask[idx][j] = 1
        # print("enc_batch维度",self.enc_batch.shape)

    def init_dec_seq(self, example_list):
        for ex in example_list:
            ex.pad_dec_inp_tgt(self.hps.max_dec_len)

        self.dec_inp_batch = np.zeros((self.bsize, self.hps.max_dec_len), dtype=np.int32)
        self.dec_tgt_batch = np.zeros((self.bsize, self.hps.max_dec_len), dtype=np.int32)
        self.dec_pad_mask = np.zeros((self.bsize, self.hps.max_dec_len), dtype=np.float32)
        self.dec_lens = np.zeros((self.bsize), dtype=np.int32)

        for idx, ex in enumerate(example_list):
            self.dec_inp_batch[idx] = ex.dec_input[:]  
            self.dec_tgt_batch[idx] = ex.dec_target[:]  
            self.dec_lens[idx] = ex.dec_len  # to calculate the posterior
            for j in range(ex.dec_len):
                self.dec_pad_mask[idx][j] = 1

    def save_original_seq(self, example_list):
        self.original_enc_text = [ex.original_enc_text for ex in example_list]
        self.original_dec_text = [ex.original_dec_text for ex in example_list]


def readJsons(filename):
	with open(filename,'r',encoding='utf8') as f:
		data = json.load(f)
	return data

def prepare_data_seq(file_path):
	claim_id = {}  
	all_id = []  
	claim_pers = {} #{cid:['claim','pers1,pers2...'],cid:[]}    

	data_ = readJsons(file_path)

	for cid in data_:
		pers = []
		claim_and_pers = []

		text = ' '.join(data_[cid]['text'])   
		# print(text)
		claim_id[cid] = text
		claim_and_pers.append(text)
		all_id.append(cid)

		for pid in data_[cid]['pers']:
			for ppid in data_[cid]['pers'][pid]:
				per = ' '.join(data_[cid]['pers'][pid][ppid])
				pers.append(per)
		claim_and_pers.append(pers)

		claim_pers[cid] = claim_and_pers

	return all_id,claim_id,claim_pers


class Batcher_:
    def __init__(self, vocab, bin_path, hps):
        assert os.path.exists(bin_path)
        self.vocab = vocab
        self.bin_path = bin_path.replace('.bin', '.json')
        self.hps = hps
        self.single_pass = True if hps.mode == 'decode' else False

    
    
    def get_claim_id(self):
        if self.hps.mode != 'lm_train':
            self.all_id, self.claim_id,self.claim_pers = prepare_data_seq(self.bin_path)
        all_id = self.all_id
        claim_id = self.claim_id
        claim_pers = self.claim_pers
        return all_id,claim_id,claim_pers
    
    def get_data_loader(self,cid,claim_id,claim_pers,b_size):   
        inputs_train = []
        inputs_val = []
        
        for idd in cid:
            dial_claim = claim_pers[idd][1]  
            # print(dial_claim)
            if(len(dial_claim))==1:
                tr = Example(claim_id[idd], dial_claim[0],  self.vocab, self.hps)
                val = tr
            elif len(dial_claim) == 2:
                tr = Example(claim_id[idd], dial_claim[0],  self.vocab, self.hps)
                val = Example(claim_id[idd], dial_claim[1],  self.vocab, self.hps)
            else:
                val_dial_num = randint(0,len(dial_claim)-1)
                val = Example(claim_id[idd], dial_claim[val_dial_num],  self.vocab, self.hps)
                tr_dial_num = randint(0,len(dial_claim)-1)
                while tr_dial_num == val_dial_num:
                    tr_dial_num = randint(0,len(dial_claim)-1)
                tr = Example(claim_id[idd], dial_claim[tr_dial_num],  self.vocab, self.hps)
            inputs_train.append(tr)
            inputs_val.append(val)
        # print(len(inputs_train))
        train_batch = Batch(inputs_train,self.hps,self.vocab,b_size)
        val_batch = Batch(inputs_val,self.hps,self.vocab,b_size)
        return train_batch,val_batch
    
    def get_val_batch(self,inputs_val_dia,b_size):
        val_bb = Batch(inputs_val_dia,self.hps,self.vocab,b_size)
        return val_bb
        

class Batcher:
    def __init__(self, vocab, bin_path, hps):
        assert os.path.exists(bin_path)
        self.vocab = vocab
        self.bin_path = bin_path
        # bin_fname = args.split_data_path.format(setname).replace('.json', '.bin')
        self.hps = hps
        self.single_pass = True if hps.mode == 'decode' else False
        if self.hps.mode != 'lm_train':
            self.all_train_example = self.read_all_sample()

        QUEUE_MAX_SIZE = 50
        self.batch_cache_size = 50
        self.batch_queue = Queue(QUEUE_MAX_SIZE)
        self.example_queue = Queue(QUEUE_MAX_SIZE * 16)#self.hps.batch_size)

        self.example_thread = Thread(target=self.fill_example_queue)
        self.example_thread.daemon = True
        self.example_thread.start()

        self.batch_thread = Thread(target=self.fill_batch_queue)
        self.batch_thread.daemon = True
        self.batch_thread.start()



    def read_all_sample(self):
        with open(self.bin_path.replace('.bin', '.json'), 'r', encoding='utf8') as f:
            data = json.load(f)
        ex_list = []
        self.cid_pers_num = {} 
        for cid in data:
            num = 0
            text = data[cid]['text']
            for pid in data[cid]['pers']:
                for ppid in data[cid]['pers'][pid]:
                    num+=1
                    ex = Example_(' '.join(text), ' '.join(data[cid]['pers'][pid][ppid]), cid, pid, ppid, self.vocab, self.hps)
                    ex_list.append(ex)
            self.cid_pers_num[cid] = num
        shuffle(ex_list)  
        # print("ex_list",ex_list)
        return ex_list  



    def next_batch(self):
        if self.batch_queue.qsize() == 0:
            if self.single_pass:
                print("[*]FINISH decoding")
                return 'FINISH'
            else:
                print("Batch queue is empty. waiting....")
            #     raise ValueError("Unexpected finish of batching.")
        batch = self.batch_queue.get()
        if self.hps.mode == 'train':
            if self.hps.model == 'mmpms':
                batch = self.fill_negative_sample(batch)
            elif self.hps.model == 'posterior':
                batch = self.fill_posneg_sample(batch)
                batch = self.fill_negative_sample(batch)

        return batch

    def fill_negative_sample(self, batch):
        """
        Randomly select the negative sample for each example and fill batch
        """
        neg_ex_list = []
        for cid in batch.cids:
            neg_ex = None
            while neg_ex is None:
                rand_ex = sample(self.all_train_example, 1)[0]
                if rand_ex.cid != cid: neg_ex = rand_ex
            neg_ex_list.append(neg_ex)

        max_enc_len = self.hps.max_dec_len
        max_gbg_len = max([ex.enc_len for ex in neg_ex_list])

        for idx,ex in enumerate(neg_ex_list):
            neg_ex_list[idx].pad_enc_input(max_gbg_len)
            neg_ex_list[idx].pad_dec_inp_tgt(max_enc_len)

        batch.neg_enc_batch = np.zeros((self.hps.batch_size, max_enc_len), dtype=np.int32)
        batch.neg_enc_lens = np.zeros((self.hps.batch_size), dtype=np.int32)
        batch.neg_enc_pad_masks =np.zeros((self.hps.batch_size, max_enc_len), dtype=np.float32)

        for idx, ex in enumerate(neg_ex_list):
            batch.neg_enc_batch[idx, :] = ex.dec_input[:]
            batch.neg_enc_lens[idx] = ex.dec_len
            for j in range(ex.dec_len):
                batch.neg_enc_pad_masks[idx][j] = 1
        return batch

    def fill_posneg_sample(self, batch):
        # Issue: How to handle if only one perspective for one claim exists?
        same_pers_ex_list = []
        diff_pers_ex_list = []

        for idx,cid in enumerate(batch.cids):
            pid, ppid = batch.pids[idx], batch.ppids[idx]
            ex_list = [ex for ex in self.all_train_example if ex.cid == cid]
            shuffle(ex_list)

            same, diff = None, None
            for ex in ex_list:
                if same is not None and diff is not None: break
                if pid == ex.pid and not ppid == ex.ppid and same is None: same = ex
                if pid != ex.pid and diff is None: diff = ex

            if same is None:
                same = Example('hi', 'hi', -1, -1, -1, voca=self.vocab, hps=self.hps)
                same.valid = False
            else:
                same.valid = True

            if diff is None:
                diff = Example('hi', 'hi', -1, -1, -1, voca=self.vocab, hps=self.hps)
                diff.valid = False
            else:
                diff.valid = True

            same_pers_ex_list.append(same)
            diff_pers_ex_list.append(diff)

        max_pos_enc_len = self.hps.max_dec_len  # max([ex.dec_len for ex in same_pers_ex_list])
        batch.pos_enc_batch = np.zeros((self.hps.batch_size, max_pos_enc_len), dtype=np.int32)
        batch.pos_enc_lens = np.zeros((self.hps.batch_size), dtype=np.int32)
        batch.pos_enc_pad_masks = np.zeros((self.hps.batch_size, max_pos_enc_len), dtype=np.float32)
        batch.pos_enc_valid = np.ones((self.hps.batch_size), dtype=np.int32)

        for idx, ex in enumerate(same_pers_ex_list):
            same_pers_ex_list[idx].pad_enc_input(max_pos_enc_len)
            same_pers_ex_list[idx].pad_dec_inp_tgt(max_pos_enc_len)

        for idx, ex in enumerate(same_pers_ex_list):
            if not ex.valid:
                batch.pos_enc_valid[idx] = 0
            batch.pos_enc_batch[idx, :] = ex.dec_input[:]
            batch.pos_enc_lens[idx] = ex.dec_len
            for j in range(ex.dec_len):
                batch.pos_enc_pad_masks[idx][j] = 1

        batch.neg_enc_valid = np.ones((self.hps.batch_size), dtype=np.int32)

        return batch

    def fill_example_queue(self):
        gen = sample_generator(self.bin_path, self.single_pass) 
        while True:
            try:
                if 'wikitext' not in self.bin_path:
                    if self.hps.model == 'mmi_bidi' and self.hps.mode == 'train':  # Reverse!
                        dec_text, enc_text, cid, pid, ppid = next(gen)
                    else:
                        enc_text, dec_text, cid, pid, ppid = next(gen)
                    example = Example(enc_text, dec_text, cid, pid, ppid, self.vocab, self.hps)
                else:
                    text = next(gen)
                    example = LMExample(text, self.vocab, self.hps)
            except Exception as err:
                print("Error while fill example queue: {}".format(self.example_queue.qsize()))
                assert self.single_pass
                break
            self.example_queue.put(example) 

    def fill_batch_queue(self):
        while True:
            if not self.single_pass:
                assert self.hps.mode != 'decode'
                inputs = []
                for _ in range(self.hps.batch_size * self.batch_cache_size):   #16*50=800
                    inputs.append(self.example_queue.get())  
                # print("inputs",inputs)

                if 'wikitext' not in self.bin_path:
                    inputs = sorted(inputs, key=lambda x: x.enc_len)  
                else:
                    inputs = sorted(inputs, key=lambda x: x.text_len)
                batches = []
                for idx in range(0, len(inputs), self.hps.batch_size): 
                    batches.append(inputs[idx:idx + self.hps.batch_size])  
                if not self.single_pass:
                    shuffle(batches)  
                for bat in batches: 
                    if 'wikitext' not in self.bin_path:
                        self.batch_queue.put(Batch(bat, self.hps, self.vocab))
                    else:
                        self.batch_queue.put(LMBatch(bat, self.hps, self.vocab))
            else:
                assert self.hps.mode == 'decode'
                sample = self.example_queue.get()
                bat = [sample for _ in range(self.hps.batch_size)]

                if 'wikitext' not in self.bin_path:
                    self.batch_queue.put(Batch(bat, self.hps, self.vocab))
                else:
                    self.batch_queue.put(LMBatch(bat, self.hps, self.vocab))


class LMExample:
    def __init__(self, text, voca, hps):
        self.hps, self.vocab = hps, voca

        toks = text.split()

        self.input_ids = self.vocab.text2ids(toks)

        self.original_text = text

        def make_dec_inp_tgt_seq(seq, max_len, beg_id, eos_id):
            inp = [beg_id] + seq[:]
            tgt = seq[:]

            if len(inp) > max_len:
                inp = inp[:max_len]
                tgt = tgt[:max_len]
            else:
                tgt.append(eos_id)
            assert len(inp) == len(tgt)
            assert len(inp) <= max_len
            assert len(tgt) <= max_len
            return inp, tgt

        self.text_input, self.text_target = make_dec_inp_tgt_seq(self.input_ids, self.hps.max_lm_len, self.vocab.beg_id,
                                                               self.vocab.eos_id)

        self.text_len = len(self.text_input)

    def pad_text_inp_tgt(self, max_len):
        assert len(self.text_input) == len(self.text_target)
        while len(self.text_input) < max_len:
            self.text_input.append(self.vocab.pad_id)
        while len(self.text_target) < max_len:
            self.text_target.append(self.vocab.pad_id)


class LMBatch():
    def __init__(self, example_list, hps, vocab):
        self.hps = hps
        self.vocab = vocab
        self.init_text_seq(example_list)
        self.save_original_seq(example_list)

    def init_text_seq(self, example_list):
        for ex in example_list:
            ex.pad_text_inp_tgt(self.hps.max_lm_len)

        self.text_lens = np.zeros((self.hps.batch_size), dtype=np.int32)
        self.text_inp_batch = np.zeros((self.hps.batch_size, self.hps.max_lm_len), dtype=np.int32)
        self.text_tgt_batch = np.zeros((self.hps.batch_size, self.hps.max_lm_len), dtype=np.int32)
        self.text_pad_mask = np.zeros((self.hps.batch_size, self.hps.max_lm_len), dtype=np.float32)

        for idx, ex in enumerate(example_list):
            self.text_inp_batch[idx] = ex.text_input[:]
            self.text_tgt_batch[idx] = ex.text_target[:]
            self.text_lens[idx] = ex.text_len
            for j in range(ex.text_len):
                self.text_pad_mask[idx][j] = 1

    def save_original_seq(self, example_list):
        self.original_text = [ex.original_text for ex in example_list]

class RvsBatch:
    def __init__(self, inp_dec_text_list, hps, vocab):

        example_list = [Example(el['tgt'],el['inp'],None,None,None,vocab,hps) for el in inp_dec_text_list]

        self.hps = hps
        self.vocab = vocab
        self.init_enc_seq(example_list)
        self.init_dec_seq(example_list)
        self.save_original_seq(example_list)

    def init_enc_seq(self, example_list):
        max_enc_len = max([ex.enc_len for ex in example_list])
        for ex in example_list:
            ex.pad_enc_input(max_enc_len)

        self.enc_batch = np.zeros((self.hps.batch_size, max_enc_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.hps.batch_size), dtype=np.int32)
        self.enc_pad_mask = np.zeros((self.hps.batch_size, max_enc_len), dtype=np.float32)

        # Fill enc batch
        for idx, ex in enumerate(example_list):
            self.enc_batch[idx, :] = ex.enc_input[:]
            self.enc_lens[idx] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_pad_mask[idx][j] = 1

    def init_dec_seq(self, example_list):
        for ex in example_list:
            ex.pad_dec_inp_tgt(self.hps.max_dec_len)

        self.dec_inp_batch = np.zeros((self.hps.batch_size, self.hps.max_dec_len), dtype=np.int32)
        self.dec_tgt_batch = np.zeros((self.hps.batch_size, self.hps.max_dec_len), dtype=np.int32)
        self.dec_pad_mask = np.zeros((self.hps.batch_size, self.hps.max_dec_len), dtype=np.float32)
        self.dec_lens = np.zeros((self.hps.batch_size), dtype=np.int32)

        for idx, ex in enumerate(example_list):
            self.dec_inp_batch[idx] = ex.dec_input[:]
            self.dec_tgt_batch[idx] = ex.dec_target[:]
            self.dec_lens[idx] = ex.dec_len  # to calculate the posterior
            for j in range(ex.dec_len):
                self.dec_pad_mask[idx][j] = 1

    def save_original_seq(self, example_list):
        self.original_enc_text = [ex.original_enc_text for ex in example_list]
        self.original_dec_text = [ex.original_dec_text for ex in example_list]
        self.cids = [ex.cid for ex in example_list]
        self.pids = [ex.pid for ex in example_list]
        self.ppids = [ex.ppid for ex in example_list]