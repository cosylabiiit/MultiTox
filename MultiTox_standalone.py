# test_new_sem8_standalone.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import esm
import pickle
import sys
import json

# -------------------------------------------------------------
# Define the base path dynamically based on the current script
# -------------------------------------------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# -------------------------------------------------------------
# Function to take input sequence(s)
# -------------------------------------------------------------
def input_toxins():
    # Read the entire standard input; may contain multiple sequences separated by newlines
    # input_data = sys.stdin.read().strip()
    # input_data = input("Enter the sequence: ")
    input_data = ["MNTKVVLIMLMITSVILVVEAETLFTANCLDRKDCKKHCKSKGCKEMKCEQIIKPTWRCLCIMCSK",
                  "MDVRFRLCLFLVILVIVANANVIKEPEKRFHPNLWRPPRCDWPHGVCSYIRDRCAPDTPFPCGPIFACPLPTNKCCCRRPYLPPWAGRR",
                  "MFKKNDRSQSRTRRHMRVRKKIFGTAERPRLSVYRSEKHIYAQLIDDVEGKTLVAASSSEKGFDGVGSNKEGAKLVGKMVAEKALEKGLKKVVFDRGGFIYHGRIKELAEGAREAGLDF",
                  "MSSLLDKTRMLNRILQKSGTEPVDFEDICDLLSDVLACNVYIISRKGKILGSKFYSGFECDEVREVVLKENRFPDFYNNKLLNVNETLSNSPNHDKCVFDNLKDCSINNKLSTIVPINGNRERLGTLLLARFDKEFTDEDLVLAEYSATIIGLEILRSKQDQIEEEARKKAVVQLAIGTLSYSELEAVEHIFNELDGTEGLLVASKIADKVGITRSVIVNALRKFESAGVIESRSLGMKGTHIRILNDKLLEELKKIK",
                  "DCAKEGEVCSWGKKCCDLDNFYCPMEFIPHCKKYKPYVPVTTNCAKEGEVCGWGSKCCHGLDCPLAFIPYCEKYRGRND",
                  "MKCATLFLVLSMVVLMAEPGDAFFHHIFRGIVHVGKTIHRLVTGGKAEQDQQDQQYQQEQQEQQAQQYQRFNRERAAFD",
                  "MDAKKMFVALVLIATFALPSLATFEKDFITPETIQAILKKSAPLSNIMLEEDVINALLKSKTVISNPIIEEAFLKNSNGLNGIPCGESCVWIPCISAAIGCSCKSKVCYRNSLDN",
                  "MPSAFEKVVKNVIKEVSGSRGDLIPVDSLRNSTSFRPYCLLNRKFSSSRFWKPRYSCVNLSIKDILEPSAPEPEPECFGSFKVSDVVDGNIQGRVMLSGMGEGKISGGAAVSDSSSASMNVCILRVTQKTWETMQHERHLQQPENKILQQLRSRGDDLFVVTEVLQTKEEVQITEVHSQEGSGQFTLPGALCLKGEGKGHQSRKKMVTIPAGSILAFRVAQLLIGSKWDILLVSDEKQRTFEPSSGDRKAVGQRHHGLNVLAALCSIGKQLSLLSDGIDEEELIEAADFQGLYAEVKACSSELESLEMELRQQILVNIGKILQDQPSMEALEASLGQGLCSGGQVEPLDGPAGCILECLVLDSGELVPELAAPIFYLLGALAVLSETQQQLLAKALETTVLSKQLELVKHVLEQSTPWQEQSSVSLPTVLLGDCWDEKNPTWVLLEECGLRLQVESPQVHWEPTSLIPTSALYASLFLLSSLGQKPC",
                  "MSNPKGTEPTILKIPTLGHAVCLGELYDQRTGNFLGVQLYPEGNIQEKETDIRHTELSLSLATSMEDKASLIDVNAKLSLEIACGLVKVSGSASYLNDSKSNTNEQAFALALKMRLNEKRILFAEEELGRNVLEVAQEDYIATGKATHFVSAIVYGGNFIVNLVAKKSKLSKEEKVEGKLKAEFSHLKGAIKLEGEVDAKIKAEFESMNDHFNLIVHGDVALEKVPINPQDVLDTFPDAASLITAGGGVPISVTLQPIPEMLIKGCLVYEIDADRVERVLEAFSLLDDLRSRFSVLTGRVKPYQDFIPELERAIRLASSQFGREHGKLLHQLRQFLYDYQHGKAPDSDIGGDVLKAAHGLYNDHLDPTKQSDPPGQYPVSLVGLETEYSTFRYLVSDIQRVRESLEPPSTKNPAQKAPKETKQSSIYLSSIEDVCRAARVQRKIPLFTMIPLAPASDEVKADLAVIQFLALIRNYGAYFKNALNPAYIVYAEVLERLERRLPNDGKFSELKHPSLFIGQVDVQGKLTWSNPPPLSSPPTSEEVQTRLSSEYPNPIVSRAMFFQSGSEPTFRFVTNRGWEFSVGAWVGPGGYSLIRRRMQSHVNFCRKDLQGTFYYKDGVFGGKLPDGQVAEALFEVLRVESKNNEPFAHIKFYVPNSTTELLGEFLARDLVGLISLRFCCGLKLCAT",
                  "LSSLPLCLPFPCFISLSWCCDTIGKNLYCVTHVGRYEQKRMTLWDRDDLENDTRERPKPNSDFEIVASESIEDKSSVLKVEASLKASFLSGLVEVEGSAKYLNDQKTSKNQARVTLTYKTTTKFKELSMNHLGRGNMKHQYVFDKGIATHVVTGILYGAQAFFVFDREGEGSLKMEDKDIANVEKFSCKFHGDFSLEKNPVTFQDAVEVYKSLPKLLGANGENAVPVKVWLLPLASLDSAAALPSIRGGGEEEAVLAEILKKRHSSPFNSEKLKEWMDCKEREICILKTFTNMMKNTKIVPSRNGVHEEILSAENAVCFAFTSLGRDEPYLSALSNYLKGTPESDDPQDPAAVDIEKQQWYASKEVADEMRKKAKLFNDFAEANKENKNVKFLTVGLTNETQKGSSIYLYKDVLSVGENFEPP",
                  "MKRILGALLGLLSAQVCCVRGIQVEQSPPDLILQEGANSTLRCNFSDSVNNLQWFHQNPWGQLINLFYIPSGTKQNGRLSATTVATERYSLLYISSSQTTDSGVYFCAVE",
                  "MLLPATMSDKPDMAEIEKFDKSKLKKTETQEKNPLPSKETIEQEKQAGES",
                  "MKYNVTMLIVFISFIPATTQAERTPNEEKKVIGYADHNGQLYNITSIYGPVINYTVPDENITINTINSTGERTQLTINYSDYVREAFNEWAPSGIRVQQVSSSGAEARVVSFSTTNYADNSLGSTIFDPSGNSRTRIDIGSFNRIVMNNFEKLKSRGAIPANMSPEEYIKLKLRITIKHEIGHILGLLHNNEGGSYFPHGVGLEVARCRLLNQAPSIMLNGSNYDYIDRLSHYLERPVTETDIGPSRNDIEGVRVMRRGGSGNSFTNRFSCLGLGLAFSRSGGDL",
                  "MDSSCHNATTKMLATAPARGNMMSTSKPLAFSIERIMARTPEPKALPVPHFLQGALPKGEPKHSLHLNSSIPCMIPFVPVAYDTSPKAGVTGSEPRKASLEAPAAPAAVPSAPAFSCSDLLNCALSLKGDLARDALPLQQYKLVRPRVVNHSSFHAMGALCYLNRGDGPCHPAAGVNIHPVASYFLSSPLHPQPKTYLAERNKLVVPAVEKYPSGVAFKDLSQAQLQHYMKESAQLLSEKIAFKTSDFSRGSPNAKPKVFTCEVCGKVFNAHYNLTRHMPVHTGARPFVCKVCGKGFRQASTLCRHKIIHTQEKPHKCNQCGKAFNRSSTLNTHTRIHAGYKPFVCEFCGKGFHQKGNYKNHKLTHSGEKQFKCNICNKAFHQVYNLTFHMHTHNDKKPFTCPTCGKGFCRNFDLKKHVRKLHDSSLGLARTPAGEPGTEPPPPLPQQPPMTLPPLQPPLPTPGPLQPGLHQGHQ",
                  "MNVFSIFSLVFLAAFGSCADDRRSALEECFREADYEEFLEIARNGLKKTSNPKHVVVVGAGMAGLSAAYVLAGAGHRVTLLEASDRVGGRVNTYRDEKEGWYVNMGPMRLPERHRIVRTYIAKFGLKLNEFFQENENAWYFIRNIRKRVWEVKKDPGVFKYPVKPSEEGKSASQLYRESLKKVIEELKRTNCSYILDKYDTYSTKEYLIKEGNLSRGAVDMIGDLLNEDSSYYLSFIESLKNDDLFSYEKRFDEISDGFDQLPKSMHQAIAEMVHLNAQVIKIQRDAEKVRVAYQTPAKTLSYVTADYVIVCATSRAVRRISFEPPLPPKKAHALRSIHYKSATKIFLTCTRKFWEADGIHGGKSTTDLPSRFIYYPNHNFTSGVGVIVAYVLADDSDFFQALDIKTSADIVINDLSLIHQLPKNEIQALCYPSLIKKWSLDKYTMGALTSFTPYQFQDYIETVAAPVGRIYFAGEYTATVHGWLDSTIKSGLTAARNVNRASQKPSRIHLINDNQL",
                  "MKFYTISSKYIEYLKEFDDKVPNSEDPTYQNPKAFIGIVLEIQGHKYLAPLTSPKKWHNNVKESSLSCFKLHENGVPENQLGLINLKFMIPIIEAEVSLLDLGNMPNTPYKRMLYKQLQFIRANSDKIASKSDTLRNLVLQGKMQGTCNFSLLEEKYRDFGKEAEDTEEGE",
                  "MLTKKELDSRKCLEYSCFEQLVPQNHLLRKIDKIIHFDFIYDEVGDLYSAVGRPSIDPIVLIKIVMIQYLFGIPFMR",
                  "MNIICIGDSLTFGYGVGQENSWVSLLNDEKNKFINKGVNGDTSTGILSRIYEILKSSDSNICLIMCGSNDILMNKSIHSIIDNIRLMTDDCNSLNIKPIILSPPKIYNDLAIKRWDSSIDYEKCNSKLENYTRELDLFCEKNNLLFIDLNSNLPFDSLNYTDGLHLSIKGNILVSNLVKTSLKNYL"]
    if isinstance(input_data, list):
        sequences = [s.strip().replace(" ", "") for s in input_data if s.strip()]
    else:
        sequences = [line.strip().replace(" ", "") for line in input_data.splitlines() if line.strip()]
    # Split input by newlines and remove extra spaces
    # sequences = [line.strip().replace(" ", "") for line in input_data.splitlines() if line.strip()]
    return sequences

# -------------------------------------------------------------
# Function to check for non-natural amino acids
# -------------------------------------------------------------
def has_non_natural(seq):
    non_natural_amino_acids = {'B', 'O', 'J', 'U', 'X', 'Z'}
    return any(aa in non_natural_amino_acids for aa in seq)

# -------------------------------------------------------------
# Function to extract ESM2 features
# -------------------------------------------------------------
def extract_esm2_features(sequence):
    # Load ESM2 model
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the sequence
    batch_tokens = torch.tensor(alphabet.encode(sequence), dtype=torch.long).unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        token_embeddings = results["representations"][33]
        seq_len = len(sequence)
        seq_embedding = token_embeddings[0, 1:seq_len + 1].mean(0).cpu().numpy()

    # Convert to DataFrame
    feature_df = pd.DataFrame([seq_embedding], columns=[str(i) for i in range(seq_embedding.shape[0])])
    return feature_df

# -------------------------------------------------------------
# Function to align features with CatBoost (or other) model
# -------------------------------------------------------------
def align_features(df_features, model_features):
    for feature in model_features:
        if feature not in df_features.columns:
            df_features[feature] = 0
    df_features = df_features[model_features]
    return df_features

def select_top_features(df_features):
    # Load the feature importances that were computed during training.
    feature_importances = np.load(os.path.join(BASE_PATH, "feature_importances.npy"))
    # We want to select the top 550 features.
    k = 550
    top_indices = np.argsort(feature_importances)[-k:]
    # print(top_indices)
    # Our df_features has column names as string representations of integer indices.
    top_columns = [str(i) for i in top_indices]
    # In case some columns are missing, fill with zeros.
    for col in top_columns:
        if col not in df_features.columns:
            df_features[col] = 0
    return df_features[top_columns]
# -------------------------------------------------------------
# Function to make final prediction with stacked approach
# -------------------------------------------------------------
def model_predict(df_features):
    try:
        # Load base models
        # model_files = ["catboost_model.pkl", "knn_model.pkl", "svc_model.pkl", "gb_model.pkl", "et_model.pkl"]
        model_files = ["lgbm_model.pkl", "knn_model.pkl", "qda_model.pkl", "et_model.pkl", "mlp_model.pkl"]
        models = {}

        for model_name in model_files:
            with open(os.path.join(BASE_PATH, model_name), 'rb') as file:
                models[model_name.split("_")[0]] = pickle.load(file)

        # First, select the top 550 features to match what the models were trained on.
        df_features = select_top_features(df_features)

        # Align features
        aligned_features = df_features.copy()
        if hasattr(models["lgbm"], 'feature_names_'):
            aligned_features = align_features(df_features, models["lgbm"].feature_names_)
        # print(aligned_features)
        # Predict probabilities from base models
        meta_features = []
        for name, clf in models.items():
            probabilities = clf.predict_proba(aligned_features)
            meta_features.append(probabilities)

        meta_features = np.hstack(meta_features)
        # print(meta_features)
        # Load meta-classifier
        with open(os.path.join(BASE_PATH, "meta_classifier.pkl"), "rb") as file:
            meta_classifier = pickle.load(file)
        # Final prediction
        print(meta_classifier.predict_proba(meta_features))
        final_prediction = meta_classifier.predict(meta_features)
        # print(final_prediction)
        return {"prediction": int(final_prediction[0])}

    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------------------
# Main function
# -------------------------------------------------------------
def main():
    all_sequences = input_toxins()
    results = []
    
    if not all_sequences:
        print(json.dumps({"error": "No input sequence provided"}))
        sys.exit(1)
    
    for seq in all_sequences:
        result = {"sequence": seq}
        if has_non_natural(seq):
            result["error"] = "Non-natural amino acids detected"
            results.append(result)
            continue

        try:
            df_features = extract_esm2_features(seq)
        except Exception as e:
            result["error"] = f"Error during feature extraction: {str(e)}"
            results.append(result)
            continue

        try:
            prediction = model_predict(df_features)
            if "error" in prediction:
                result["error"] = prediction["error"]
            else:
                result["prediction"] = int(prediction["prediction"])
        except Exception as e:
            result["error"] = f"Error during prediction: {str(e)}"
        results.append(result)
    
    print(json.dumps({"results": results}))

# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------
if __name__ == "__main__":
    main()



