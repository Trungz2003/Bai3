package org.example;

import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Attribute;

import java.util.ArrayList;

public class DatasetBuilder {
    public static Instances createDataset() {
        // Các thuộc tính cho tập dữ liệu
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < 256; i++) {
            attributes.add(new Attribute("histogram_" + i));
        }
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("flower");
        classValues.add("animal");
        attributes.add(new Attribute("class", classValues));

        // Tạo tập dữ liệu
        Instances dataset = new Instances("ImageDataset", attributes, 0);
        dataset.setClassIndex(attributes.size() - 1); // Đặt thuộc tính lớp là chỉ số cuối cùng

        return dataset;
    }

    public static void addInstance(Instances dataset, double[] features, String classValue) {
        // Tạo một instance mới với số lượng thuộc tính của tập dữ liệu
        DenseInstance instance = new DenseInstance(dataset.numAttributes());

        // Gán dataset cho instance trước khi thiết lập giá trị thuộc tính
        instance.setDataset(dataset);

        // Thiết lập giá trị thuộc tính cho instance
        for (int i = 0; i < features.length; i++) {
            instance.setValue(i, features[i]);
        }

        // Thiết lập lớp cho instance
        instance.setClassValue(classValue);

        // Thêm instance vào dataset
        dataset.add(instance);
    }
}


