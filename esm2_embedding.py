
# esm2_extractor.py
# Usage: python esm2_extractor.py --input toxin_3052.csv --output toxin3052_ESM_embeddings.csv.csv

import pandas as pd
import torch
import esm
import argparse
from tqdm import tqdm

def load_data(input_path):
    df = pd.read_csv(input_path)
    df = df[['Sequence', 'Label']].dropna()
    df = df[df['Sequence'].apply(lambda x: isinstance(x, str) and len(x) > 20)].reset_index(drop=True)
    return df

def extract_embeddings(seqs, model, batch_converter, device, batch_size=4):
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), batch_size)):
            batch_seqs = seqs[i:i+batch_size]
            batch_labels = [f"seq{i+j}" for j in range(len(batch_seqs))]
            batch_data = list(zip(batch_labels, batch_seqs))
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)

            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

            for j, (_, seq) in enumerate(batch_data):
                # Exclude CLS and EOS tokens
                representation = token_representations[j, 1:len(seq)+1].mean(0)
                all_embeddings.append(representation.cpu().numpy())

    return torch.tensor(all_embeddings)

def main():
    parser = argparse.ArgumentParser(description="Extract ESM2 embeddings from a CSV file.")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV with Sequence and Label columns.")
    parser.add_argument('--output', type=str, required=True, help="Path to save output CSV with embeddings and labels.")
    args = parser.parse_args()

    print("🔹 Loading model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    print("🔹 Loading data...")
    df = load_data(args.input)
    sequences = df["Sequence"].tolist()

    print("🔹 Extracting embeddings...")
    embeddings = extract_embeddings(sequences, model, batch_converter, device, batch_size=4)

    print("🔹 Saving output...")
    embedding_df = pd.DataFrame(embeddings.numpy())
    embedding_df["Label"] = df["Label"]
    embedding_df.to_csv(args.output, index=False)
    print(f"✅ Embeddings saved to: {args.output}")

if __name__ == "__main__":
    main()
