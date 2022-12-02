package com.example.finalproject.model;

import androidx.annotation.NonNull;

public class NamedEntity {
    private final String type;
    private final String text;

    public NamedEntity(String type, String text) {
        this.type = type;
        this.text = text;
    }

    public String getType() {
        return type;
    }

    public String getText() {
        return text;
    }

    @NonNull
    @Override
    public String toString() {
        return "NamedEntity{" +
                "type='" + type + '\'' +
                ", text='" + text + '\'' +
                '}';
    }
}
