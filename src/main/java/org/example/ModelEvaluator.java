package org.example;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Utils;

public class ModelEvaluator {
    public static void evaluateModel(Classifier classifier, Instances dataset) throws Exception {
        dataset.setClassIndex(dataset.numAttributes() - 1); // Đặt chỉ số lớp
        double correct = 0;

        // Lặp qua tất cả các instance để đánh giá
        for (int i = 0; i < dataset.numInstances(); i++) {
            double predicted = classifier.classifyInstance(dataset.instance(i));
            if (predicted == dataset.instance(i).classValue()) {
                correct++;
            }
        }

        double accuracy = correct / dataset.numInstances() * 100.0;
        System.out.println("Accuracy: " + Utils.doubleToString(accuracy, 2) + "%");
    }
}
