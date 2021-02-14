from util.dataset import load_mat, train_test_split
import sklearn
from sklearn import svm
import pickle


jump = 1
temp_dir = "./dataset/SEED/ExtractedFeatures/de_LDS"
feature, label, cumulative = load_mat(temp_dir)
train_arr, train_label, test_arr, test_label = train_test_split(feature, label, cumulative)
train_arr = train_arr[::jump]
train_label = train_label[::jump]
test_arr = test_arr[::jump]
test_label = test_label[::jump]
print("Train phase:")
classifier = svm.SVC()
classifier.fit(train_arr, train_label.ravel())
# fd_classifier = open("classifier_svm", "rb")
# classifier = pickle.load(fd_classifier)
fd_classifier = open("classifier_svm", "wb")
pickle.dump(classifier, fd_classifier)
print("Predict phase:")
predict_label = classifier.predict(test_arr)
score = sklearn.metrics.accuracy_score(test_label.ravel(), predict_label)
print("score: ", score)