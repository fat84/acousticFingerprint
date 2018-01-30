
## List comprehension is:

# [leaf for branch in tree for leaf in branch]

# for branch in tree:
#    for leaf in branch:
#        yield leaf



## In an example, the above statement looks like this:

# this is the same as ....
print([x for x in songs_list if x.find("wav") < 0])

# ... this
for x in songs_list:
    if x.find("wav") < 0:
        print(x)




