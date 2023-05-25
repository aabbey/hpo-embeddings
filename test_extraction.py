import get_hpo_terms
import conversation_direct


def create_confusion_matrix(predicted_set, actual_set):
    TP = len(predicted_set.intersection(actual_set))  # True Positive
    FP = len(predicted_set.difference(actual_set))  # False Positive
    FN = len(actual_set.difference(predicted_set))  # False Negative
    TN = len((predicted_set.union(actual_set)).difference(predicted_set.intersection(actual_set)))  # True Negative

    # creating the confusion matrix
    confusion_matrix = [
        ['Predicted/Actual', 'Positive', 'Negative'],
        ['Positive', TP, FP],
        ['Negative', FN, TN]
    ]

    # printing the confusion matrix
    for row in confusion_matrix:
        print("{:<15} {:<15} {:<15}".format(*row))



if __name__ == "__main__":
    true_terms1 = {'Sleep-wake cycle disturbance', 'Cerebral cortical atrophy', 'Hypoplasia of the maxilla',
                  'Limb tremor', 'Fair hair', 'EEG abnormality', 'Constipation', 'Mandibular prognathia', 'Strabismus',
                  'Absent speech', 'Clumsiness', 'Deeply set eye', 'Protruding tongue', 'Hypopigmentation of the skin',
                  'Feeding difficulties in infancy', 'Global developmental delay', 'Blue irides',
                  'Generalized hypotonia', 'Macroglossia', 'Sporadic'}
    true_terms = {'Hypotonia', 'Deeply set eye', 'EEG abnormality', 'Sleep-wake cycle disturbance', 'Drooling', 'Mandibular prognathia', 'Intellectual disability, severe', 'Blue irides', 'Wide mouth', 'Scoliosis', 'Clumsiness', 'Hyperactivity', 'Hypoplasia of the maxilla', 'Sporadic', 'Cerebral cortical atrophy', 'Intellectual disability, progressive', 'Strabismus', 'Progressive gait ataxia', 'Nystagmus', 'Autosomal dominant inheritance'}

    extracted_terms = get_hpo_terms.extract_terms(conversation_path="sample generated transcripts/gpt4-sample1-angelman")

    print(extracted_terms)
    create_confusion_matrix(extracted_terms, true_terms)
