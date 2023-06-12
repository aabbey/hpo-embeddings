import extract
import conversation_direct
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles
import json


sample_transcript = "sample generated transcripts/gpt4-sample1-Biotinidase.json"


def compare_sets(set1, set2):
    plt.figure(figsize=(10, 7))
    venn = venn2([set(set1), set(set2)], set_labels=('Extracted Terms', 'True Terms'))

    venn.get_label_by_id('10').set_text('\n'.join(map(str, set1 - set2)))
    venn.get_label_by_id('01').set_text('\n'.join(map(str, set2 - set1)))
    venn.get_label_by_id('11').set_text('\n'.join(map(str, set1 & set2)))

    venn2_circles([set(set1), set(set2)])
    plt.show()


if __name__ == "__main__":
    with open(sample_transcript, 'r') as json_file:
        data = json.load(json_file)

    transcript = data['transcript']
    true_terms = set(data['true_terms'])

    extracted_terms = get_hpo_terms.extract_terms(transcript=transcript, verbose=False)

    print("Extracted terms : ", extracted_terms)
    compare_sets(extracted_terms, true_terms)
