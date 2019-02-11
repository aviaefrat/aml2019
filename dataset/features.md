### character based
#### unicode block
get the proportion of characters in each unicode block
more specifically, group most uncommon blocks together, to reduce amount of features
can refine this feature by ignoring character originating from retweets, links, hashtag, etc.

#### character proportion
get the proportion of characters
We can't handle all possible characters, so we need to decide which to treat independently, and which
to group together as others (probably all the latin ones should be distinct features, but I need to a. 
verify that it includes accents, and b. check the data to see other char nominees). Maybe even deduce
the features from the training data, say use every char that is more than 0.01% or something in the
total in at least one label. 