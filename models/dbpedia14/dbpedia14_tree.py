from treelib import Tree, Node

tree = Tree()

root = tree.create_node("Entity", "Entity")
print(tree.show(stdout=False))

tree.create_node(":Agent", ":Agent", parent="owl:Thing")
tree.create_node(":Organisation", ":Organisation", parent=":Agent")
tree.create_node(":EducationalInstitution", ":EducationalInstitution", parent=":Organisation")
tree.create_node(":Company", ":Company", parent=":Organisation")

tree.create_node(":Species", ":Species", parent="owl:Thing")
tree.create_node(":Eukaryote", ":Eukaryote", parent=":Species")
tree.create_node(":Animal", ":Animal", parent=":Eukaryote")
tree.create_node(":Person", ":Person", parent=":Animal")
tree.create_node(":OfficeHolder", ":OfficeHolder", parent=":Person")
tree.create_node(":Athlete", ":Athlete", parent=":Person")
tree.create_node(":Artist", ":Artist", parent=":Person")

tree.create_node(":Plant", ":Plant", parent=":Eukaryote")

tree.create_node(":MeansOfTransport", ":MeansOfTransport", parent="owl:Thing")

tree.create_node(":Work", ":Work", parent="owl:Thing")
tree.create_node(":Album", ":Album", parent=":Work")
tree.create_node(":Film", ":Film", parent=":Work")
tree.create_node(":WrittenWork", ":WrittenWork", parent=":Work")

tree.create_node(":Settlement", ":Settlement", parent="owl:Thing")
tree.create_node(":Village", ":Village", parent=":Settlement")

tree.create_node(":Place", ":Place", parent="owl:Thing")
tree.create_node(":NaturalPlace", ":NaturalPlace", parent=":Place")

tree.create_node(":ArchitecturalStructure", ":ArchitecturalStructure", parent="owl:Thing")
tree.create_node(":Building", ":Building", parent=":ArchitecturalStructure")

print(tree.show(stdout=False))