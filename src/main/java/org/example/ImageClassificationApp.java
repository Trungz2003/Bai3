package org.example;

import weka.core.Instances;
import java.io.File;

public class ImageClassificationApp {
    public static void main(String[] args) throws Exception {
        // Tạo tập dữ liệu
        Instances dataset = DatasetBuilder.createDataset();

        // Thêm các mẫu vào tập dữ liệu
        File flowerImage = new File("src/main/java/org/example/img/a2.png");
        double[] flowerFeatures = ImageFeatureExtractor.extractColorHistogram(flowerImage);
        DatasetBuilder.addInstance(dataset, flowerFeatures, "flower");

        File animalImage = new File("src/main/java/org/example/img/a1.png");
        double[] animalFeatures = ImageFeatureExtractor.extractColorHistogram(animalImage);
        DatasetBuilder.addInstance(dataset, animalFeatures, "animal");

        // Huấn luyện mô hình KNN
        KNNClassifier knnClassifier = new KNNClassifier();
        knnClassifier.buildClassifier(dataset); // Sử dụng buildClassifier thay vì train()

        // Đánh giá mô hình
        ModelEvaluator.evaluateModel(knnClassifier, dataset);

        // Phân loại ảnh mới
        File testImage1 = new File("src/main/java/org/example/img/a1.png"); // Đường dẫn đến ảnh cần phân loại
        double[] testFeatures1 = ImageFeatureExtractor.extractColorHistogram(testImage1);
        String predictedClass1 = knnClassifier.classify(testFeatures1);
        System.out.println("Predicted class for test image a1: " + predictedClass1);

        // Phân loại ảnh mới
        File testImage = new File("src/main/java/org/example/img/a2.png"); // Đường dẫn đến ảnh cần phân loại
        double[] testFeatures = ImageFeatureExtractor.extractColorHistogram(testImage);
        String predictedClass = knnClassifier.classify(testFeatures);
        System.out.println("Predicted class for test image a2: " + predictedClass);
    }
}
