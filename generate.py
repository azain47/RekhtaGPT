from gpt2 import GPT2, GPTConfig 
import torch
import torch.nn.functional as F
import sentencepiece

initial_sentence = ""

max_length = 450

temperature = 0.8

p = 0.8

repeat_samples = 3



# ------------------------------------------------------------------------
config = GPTConfig(
    block_size=512,
    vocab_size=8192,
    n_layer=12,
    n_head=8,
    n_embed=768,
    use_rotary=True
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict = torch.load('RekhtaGPT.pt')
model = GPT2(config).to(device=device)
model.load_state_dict(state_dict)

model.eval()
enc = sentencepiece.SentencePieceProcessor()
enc.Load('./tokenizer.model')
tokens = enc.Encode(initial_sentence, add_bos=True)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(repeat_samples,1)
x = tokens.to('cuda')


# seed = torch.Generator(device='cuda').seed()
# torch.manual_seed(seed)
# Top-P generation
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        # choose last token
        logits = logits[:,-1,:]

        probs = F.softmax(logits / temperature, dim=-1)
        
        # sort the probs, and calculate cumulative sum.
        probs_sort, probs_idx = torch.sort(probs,dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # the mask tells us the probabilities 
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0
        probs_sort.div_(probs_sort.sum(dim = -1, keepdim = True))
        ix = torch.multinomial(probs_sort, 1)

        xcol = torch.gather(probs_idx, -1, ix)
        if xcol[0].tolist()[0] == enc.eos_id():
            break
        x = torch.cat((x,xcol), dim=1)

for i in range(repeat_samples):
    
    try:
        eos = x[i].tolist().index(enc.eos_id())
    except:
        eos = len(x[i])

    decoded = enc.DecodeIds(x[i].tolist()[:eos]).replace('[<n>]','\n')
    print(f'Top-P Generation {i+1}:\n{decoded}')

# x = tokens.to('cuda')
# # Top-K generation
# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits = model(x)

#         logits = logits[:,-1,:]

#         probs = F.softmax(logits / temperature, dim=-1)
        
#         topk_probs, topk_indices = torch.topk(probs, 50)

#         ix = torch.multinomial(topk_probs, 1)

#         xcol = torch.gather(topk_indices, -1, ix)
#         if xcol[0].tolist()[0] == 3:
#             break
#         x = torch.cat((x,xcol), dim=1)

# for i in range(2):
#     decoded = enc.DecodeIds(x[i].tolist()).replace('[<n>]','\n')
#     print(f'Top-K Generation {i+1} Seed {seed} :\n{decoded}')
