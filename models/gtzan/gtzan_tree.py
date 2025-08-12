from treelib import Tree, Node

tree = Tree()

root = tree.create_node("owl:Thing", "owl:Thing")
print(tree.show(stdout=False))

tree.create_node(":electronic_rock", ":electronic_rock", parent="owl:Thing")
tree.create_node(":jazz_blues_reggae", ":jazz_blues_reggae", parent="owl:Thing")
tree.create_node(":POP", ":POP", parent="owl:Thing")
tree.create_node(":classical_country", ":classical_country", parent="owl:Thing")

tree.create_node(":disco", ":disco", parent=":electronic_rock")
tree.create_node(":rock", ":rock", parent=":electronic_rock")
tree.create_node(":metal", ":metal", parent=":electronic_rock")

tree.create_node(":jazz", ":jazz", parent=":jazz_blues_reggae")
tree.create_node(":blues", ":blues", parent=":jazz_blues_reggae")
tree.create_node(":reggae", ":reggae", parent=":jazz_blues_reggae")

tree.create_node(":pop", ":pop", parent=":POP")
tree.create_node(":hiphop", ":hiphop", parent=":POP")

tree.create_node(":classical", ":classical", parent=":classical_country")
tree.create_node(":country", ":country", parent=":classical_country")

print(tree.show(stdout=False))


list = [
    
]