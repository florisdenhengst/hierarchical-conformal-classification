nodes = ['entity', 'electronic_rock', 'jazz_blues_reggae', 'POP', 'classical_country', 'disco', 'rock', 'metal', 'jazz', 'blues', 'reggae', 'pop', 'hiphop', 'classical', 'country']

template = """  <LexicalEntry id="genre-en-{}-n">
        <Lemma writtenForm="{}" partOfSpeech="n" />
        <Sense id="genre-en-{}-sense-{}-n" synset="genre-en-{}-synset-{}-n" />
    </LexicalEntry>
"""

for i, n in enumerate(nodes):
    print(template.format(n, n, n, 0, n, 0))