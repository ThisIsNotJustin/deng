import os
import torch
from deng import Transformer, init_tokenizer, device, get_tokenizer

def load_best_model(model, checkpoint_dir='checkpoint'):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        return
    
    best = sorted(checkpoints, key=lambda x: float(x.split('_loss_')[1].split('.pt')[0]))[0]
    checkpoint_path = os.path.join(checkpoint_dir, best)

    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"Failed loading model {e}")
        return False

def translate(model, tokenizer, src_sentence, direction='en-nl', max_length=50):
    model.eval()
    src_text = f'<2nl> {src_sentence}' if direction == 'en-nl' else f'<2en> {src_sentence}'
    src_ids = tokenizer.encode(src_text).ids
    src = torch.tensor(src_ids).unsqueeze(0).to(device)
    
    trg_ids = [tokenizer.token_to_id('<sos>')]
    for _ in range(max_length):
        trg = torch.tensor(trg_ids).unsqueeze(0).to(device)
        output = model(src, trg)
        next_id = output.argmax(2)[:, -1].item()
        trg_ids.append(next_id)

        if next_id == tokenizer.token_to_id('<eos>'):
            break
    
    translation = tokenizer.decode(trg_ids, skip_special_tokens=True)
    translation = translation.replace('Ġ', '')
    return translation.strip()

def main():
    init_tokenizer('eng-dutch.tsv')
    tokenizer = get_tokenizer()

    vocab_size = tokenizer.get_vocab_size()
    model = Transformer(vocab_size).to(device)
    model.eval()

    if not load_best_model(model):
        return
    
    test = [
        "Hello world!",
        "How are you?",
        "Good morning!",
        "Goodnight!",
    ]

    print("\nTesting English to Dutch:")
    for sentence in test:
        translation = translate(model, tokenizer, sentence, direction='en-nl')
        print(f"EN: {sentence}")
        print(f"NL: {translation}\n")

if __name__ == '__main__':
    main()