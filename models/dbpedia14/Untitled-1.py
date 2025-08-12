
classes = [
    "eukaryote",
    "animal",
    "person",
    "artist",
    "athlete",
    "officeholder",
    "plant",
    "work",
    "album",
    "film",
    "writtenwork",
]

TEMPLATE = """
    <LexicalEntry id="dbpedia14-en-{}-n">
        <Lemma writtenForm="{}" partOfSpeech="n" />
        <Sense id="dbpedia14-en-{}-sense-0-n" synset="dbpedia14-en-{}-synset-0-n" />
    </LexicalEntry>
    """
TEMPLATE = """
    <Synset id="dbpedia14-en-{}-synset-0-n" ili="in" partOfSpeech="n" members="dbpedia14-en-{}-sense-0-n">
        <SynsetRelation relType="hypernym" target="dbpedia14-en-12345-synset-0-n" />
    </Synset>
    """

for c in classes:
    print(TEMPLATE.format(*[c,]*2))