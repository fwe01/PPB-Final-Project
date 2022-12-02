package com.example.finalproject.adapter;

import android.content.Intent;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.example.finalproject.R;
import com.example.finalproject.model.NamedEntity;

import java.util.ArrayList;

public class NamedEntityAdapter extends RecyclerView.Adapter<NamedEntityAdapter.ViewHolder> {
    private ArrayList<NamedEntity> namedEntities;

    public NamedEntityAdapter(ArrayList<NamedEntity> namedEntities) {
        this.namedEntities = namedEntities;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View itemView = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.named_entity_list_item, parent, false);
        return new ViewHolder(itemView);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        NamedEntity namedEntity = namedEntities.get(position);
        holder.button.setText(namedEntity.getText());
        holder.button.setOnClickListener(view -> {
            System.out.println("opening google for " + namedEntity.getText());
            Intent myIntent = new Intent(Intent.ACTION_VIEW, Uri.parse("http://www.google.com"));
            view.getContext().startActivity(myIntent);
        });
    }

    @Override
    public int getItemCount() {
        return namedEntities.size();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        Button button;

        ViewHolder(View view) {
            super(view);
            button = view.findViewById(R.id.btn_named_entity_item);
        }
    }

}
