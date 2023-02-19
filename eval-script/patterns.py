from dataset import ANTONYM, COHYPONYM, HYPERNYM, CORRUPTION, RELATIONS, AnnotatedWord
import mlm

# TODO [MASK] -> tokenizer mask token!!

WORD_TOKEN = '<W>'
MASK_TOKEN = '[MASK]'

def get_patterns(word: AnnotatedWord, relation: str, model: mlm.AbstractMaskedLanguageModel):
    if model.model_str in ['roberta', 'bert']:
        lang = 'en'
    elif model.model_str in ['camembert']:
        lang = 'fr'
    elif model.model_str in ['herbert']:
        lang = 'pl'
    else:
        raise ValueError("Unknown model {}".format(model.model_str))

    if relation == ANTONYM:
        return get_patterns_antonym(word, lang)
    if relation == COHYPONYM:
        return get_patterns_cohyponym(word, lang)
    if relation == HYPERNYM:
        return get_patterns_hypernym(word, lang)
    if relation == CORRUPTION:
        return get_patterns_corruption(word, lang)
    raise ValueError("No patterns found for relation {}".format(relation))


def get_patterns_antonym(_, lang):
    return {
            'en': [
                '<W> is the opposite of [MASK]',
                '<W> is not [MASK]',
                'someone who is <W> is not [MASK]',
                'something that is <W> is not [MASK]',
                '" <W> " is the opposite of " [MASK] "'
            ],
            'fr': [
                '<W> est le contraire de [MASK]',
                '<W> n\'est pas [MASK]',
                'quelqu\'un qui est <W> n\'est pas [MASK]',
                'quelque chose qui est <W> n\'est pas [MASK]',
                '" <W> " est le contraire de " [MASK] "'
            ],
            'pl': [
                '<W> to przeciwieństwo do słowa [MASK]',
                '<W> to nie [MASK]',
                'ktoś <W> nie jest [MASK]',
                'obiekt <W> nie jest [MASK]',
                '" <W> " jest przeciwieństwem do słowa " [MASK] "'
            ]
    }[lang]


def get_patterns_hypernym(word, lang):
    article = _get_article(word, lang)

    return {
            'en': [
                '<W> is a [MASK]',
                '<W> is an [MASK]',
                article + ' <W> is a [MASK]',
                article + ' <W> is an [MASK]',
                '" <W> " refers to a [MASK]',
                '" <W> " refers to an [MASK]',
                '<W> is a kind of [MASK]',
                article + ' <W> is a kind of [MASK]'
            ],
            'fr': [
                '<W> est un [MASK]',
                '<W> est une [MASK]',
                article + ' <W> est un [MASK]',
                article + ' <W> est une [MASK]',
                '" <W> " fait référence à un [MASK]',
                '" <W> " fait référence à une [MASK]',
                '<W> est un type de [MASK]',
                article + ' <W> est un type de [MASK]'
            ],
            'pl': [
                '<W> to [MASK]',
                '" <W> " nawiązuje do słowa [MASK]',
                '<W> jest rodzajem czegoś co nazywa się [MASK]',
                'Element kategorii do której należy <W> to [MASK]',
            ]
    }[lang]


def get_patterns_cohyponym(_, lang):
    return {
            'en': [
                '<W> and [MASK]',
                '" <W> " and " [MASK] "'
            ],
            'fr': [
                '<W> et [MASK]',
                '" <W> " et " [MASK] "'
            ],
            'pl': [
                '<W> i [MASK]',
                '" <W> " i " [MASK] "'
            ]
    }[lang]


def get_patterns_corruption(_, lang):
    return {
            'en': [
                '" <W> " is a misspelling of " [MASK] " .',
                '" <W> " . did you mean " [MASK] " ?'
            ],
            'fr': [
                '" <W> " est une version mal orthographiée de " [MASK] " .',
                '" <W> " . vouliez-vous écrire " [MASK] " ?'
            ],
            'pl': [
                '" <W> " to literówka słowa " [MASK] " .',
                '" <W> " . masz na myśli " [MASK] " ?'
            ]
    }[lang]


def _get_article(word, lang):
    if lang == 'en':
        if word.word[0] in ['a', 'e', 'i', 'o', 'u']:
            return 'an'
        return 'a'
    if lang == 'fr':
        return 'un(e)'
    if lang == 'pl':
        return ''


if __name__ == '__main__':

    dummy = AnnotatedWord('dummy', pos='n', freq=1, count=1)

    for rel in RELATIONS:
        try:
            print('=== {} patterns ({}) ==='.format(rel, len(get_patterns(dummy, rel))))
            for p in get_patterns(dummy, rel):
                print(p)
        except ValueError:
            print('=== no patterns found for relation {} ==='.format(rel))
        print('')
