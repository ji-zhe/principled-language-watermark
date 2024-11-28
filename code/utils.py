import torch
import numpy as np
import torch.nn.functional as F

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def generate_msgs(args):
    msgs_num = args.msgs_num
    msgs = np.random.choice([0,1], [msgs_num,args.msg_len])
    np.savetxt('msgs.txt',msgs.astype(int))
    return msgs

def word_substitute(tokens, x, p):     # substitute words with probability p
    keep = (torch.rand(x.size(), device=x.device) > p)
    x_ = x.clone()
    x_.random_(0, tokens)
    x_[keep] = x[keep]
    return x_

def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def get_batch_no_msg(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def get_batch_noise(source, i, args, tokens, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    data_noise = word_substitute(tokens, data, args.sub_prob)
    target = source[i+1:i+1+seq_len].view(-1)
    return data, data_noise, target


def get_batch_different(source, i, args, all_msgs, seq_len=None, evaluation=False):
    # get a different random msg for each sentence.
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    bsz = data.size(1)
    msg = np.random.choice([0,1], [bsz,args.msg_len])
    msg = torch.from_numpy(msg).float()
    if args.cuda:
        msg = msg.cuda()
    return data, msg, target

def calculate_label(msg):
    label = 0
    for i in range(msg.shape[0]):
        label += msg[i] * (2**(msg.shape[0]-i-1))
    return label

def get_batch_different_cgan(source, i, args, all_msgs, seq_len=None, evaluation=False):
    # get a different random msg for each sentence.
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    bsz = data.size(1)
    msg = np.random.choice([0,1], [bsz,args.msg_len])
    # msg_label = np.array(list(map(calculate_label, msg)))
    # msg_label = torch.from_numpy(msg_label)
    # msg_label = F.one_hot(msg_label, num_classes = 2 ** args.msg_len)
    msg = torch.from_numpy(msg).float()
    # msg_label = msg_label.float()
    labels = list(map(calculate_label, msg))
    labels = torch.Tensor(labels).to(torch.int64)
    labels = F.one_hot(labels, num_classes = 2 ** args.msg_len).float()
    if args.cuda:
        msg = msg.cuda()
        labels = labels.cuda()
    return data, msg, target, labels
    
def batchify_test(data, bsz, args_cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args_cuda:
        data = data.cuda()
    return data

def generate_msgs_test(msgs_num,msg_len):
    msgs = np.random.choice([0,1], [msgs_num,msg_len])
    np.savetxt('msgs.txt',msgs.astype(int))
    return msgs

def freeze(m):
    for p in m.parameters():
        p.requires_grad_(False)
def unfreeze(m):
    for p in m.parameters():
        p.requires_grad_(True)

def distance_seq2seq(seq1, seq2):
    # (S,N,E) distance
    feature1 = seq1.mean(dim=0)
    feature2 = seq2.mean(dim=0)
    return torch.norm(feature1-feature2,dim=1)
