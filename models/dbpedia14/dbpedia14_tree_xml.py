import xml.etree.ElementTree as et
from io import StringIO, BytesIO

# WORDNET_DOCTYPE = '<!DOCTYPE LexicalResource SYSTEM "http://globalwordnet.github.io/schemas/WN-LMF-1.3.dtd">'

document = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE LexicalResource SYSTEM "http://globalwordnet.github.io/schemas/WN-LMF-1.3.dtd">
<LexicalResource xmlns:dc="http://purl.org/dc/elements/1.1/">
<a>test</a>
</LexicalResource>
"""
root = et.Element("LexicalResource")

root = tree.getroot()

f = BytesIO()
tree.write(f,
           encoding='utf-8',
           xml_declaration=True,
)
print("f.getvalue())
