Data mining project for amazon's datasets. Work in progress

Μπαρμπαρόσος Θεοφάνης: 1115202200107

Τα αρχεία part1.ipynb και part2.ipynb είναι η τελευταία τρεγμένη έκδοση των notebooks μας και για τα δύο parts.
Τα αποτελέσματα που περιέχουν, αφορούν το μέγιστο μέγεθος των datasetsμε τα οποία μπορούσαμε να τα τρέξουμε τοπικά.
Σχεδιαστικές επιλογές, επεξηγήσεις και σχολιασμός για τα notebooks παρέχονται μέσα στα ίδια τα notebooks.


timespan: May'96 - Sep'23


Ο λόγος που έχουμε κάποιο overlap στο clustering και όχι ιδιαίτερα ψηλό silhouette score είναι επειδή
είναι nlp task και το κείμενο πολύ συχνά κάνει overlap σε νοήματα κλπ

Ο λόλος που το k range ξεκινάει από 5 είναι επειδή παρατηρήθηκε ότι αν ξεκινούσαμε κάτω από 5 τότε δύο κατηγορίες
έβρισκαν βέλτιστο k το 4 και κάποια clusters ήταν σχεδόν άδεια που σημαίνει ότι το ιδανικό θα ήταν είτε
λιγότερα clusters(που δεν θα είχε νόημα) είτε περισσότερα. Οπότε ορίζοντας υποχρεωτικά κατ'ελάχιστο 5 clusters
πετυχαίνουμε καλό αποτέλεσμα clustering


todo:
fix recall scores in recommendation system
fix reviewing results of recommendation system
add comments and textual analysis of the results
