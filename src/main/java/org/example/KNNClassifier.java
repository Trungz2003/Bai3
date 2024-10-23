package org.example;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Capabilities;

public class KNNClassifier implements Classifier {
    private IBk knn;
    private Instances trainingData; // Khai báo trainingData

    public KNNClassifier() {
        knn = new IBk(); // Khởi tạo mô hình KNN
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        this.trainingData = data; // Lưu lại tập dữ liệu huấn luyện
        knn.buildClassifier(data); // Huấn luyện mô hình KNN
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return knn.classifyInstance(instance); // Phân loại
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return knn.distributionForInstance(instance); // Phân phối nhãn cho instance
    }

    @Override
    public Capabilities getCapabilities() {
        return knn.getCapabilities(); // Trả về khả năng của mô hình KNN
    }

    public String classify(double[] featureVector) throws Exception {
        DenseInstance instance = new DenseInstance(1.0, featureVector);
        instance.setDataset(trainingData); // Thiết lập dataset cho instance

        double result = classifyInstance(instance);
        return trainingData.classAttribute().value((int) result); // Trả về nhãn đã phân loại
    }
}
