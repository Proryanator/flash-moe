import json, struct, os

tok_path = os.path.expanduser('/Users/ryan/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3/tokenizer.json')
with open(tok_path, 'r', encoding='utf-8') as f:
    t = json.load(f)

vocab = t['model']['vocab']
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

# Build byte-level BPE decode table (GPT-style)
bs = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256+n)
        n += 1
byte_decoder = {chr(c): bytes([b]) for b, c in zip(bs, cs)}

with open('vocab.bin', 'wb') as f:
    f.write(struct.pack('<I', len(sorted_vocab)))
    f.write(struct.pack('<I', sorted_vocab[-1][1]))
    for token_str, token_id in sorted_vocab:
        try:
            decoded = b''.join(byte_decoder.get(c, c.encode('utf-8')) for c in token_str)
        except:
            decoded = token_str.encode('utf-8')
        f.write(struct.pack('<H', len(decoded)))
        f.write(decoded)

print(f'Rebuilt vocab.bin: {len(sorted_vocab)} entries (byte-level BPE decoded)')