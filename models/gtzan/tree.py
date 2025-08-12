from treelib import Tree, Node

tree = Tree()

root = tree.create_node("Entity", "Entity")
print(tree.show(stdout=False))

tree.create_node(":Electronic/Rock", ":Electronic/Rock", parent="owl:Thing")
tree.create_node(":Disco", ":Disco", parent=":Electronic/Rock")
tree.create_node(":Rock", ":Rock", parent=":Electronic/Rock")
tree.create_node(":Metal", ":Metal", parent=":Electronic/Rock")

tree.create_node(":Jazz/Blues/Reggae", ":Jazz/Blues/Reggae", parent="owl:Thing")
tree.create_node(":Jazz", ":Jazz", parent=":Jazz/Blues/Reggae")
tree.create_node(":Blues", ":Blues", parent=":Jazz/Blues/Reggae")
tree.create_node(":Reggae", ":Reggae", parent=":Jazz/Blues/Reggae")

tree.create_node(":POP", ":POP", parent="owl:Thing")
tree.create_node(":Pop", ":Pop", parent=":POP")
tree.create_node(":Hiphop", ":Hiphop", parent=":POP")

tree.create_node(":Classical/Country", ":Classical/Country", parent="owl:Thing")
tree.create_node(":Classical", ":Classical", parent=":Classical/Country")
tree.create_node(":Country", ":Country", parent=":Classical/Country")

print(tree.show(stdout=False))