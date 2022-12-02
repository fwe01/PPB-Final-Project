package com.example.finalproject;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;

import androidx.navigation.ui.AppBarConfiguration;

import com.example.finalproject.databinding.ActivityMainBinding;

import android.view.MenuItem;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Objects;

public class MainActivity extends AppCompatActivity {
    private static final int MODEL_INPUT_LENGTH = 512;
    private static final int NUM_LABEL = 39;
    public static final int BEGIN_LABEL_COUNT = 18;

    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;

    private final String CLS = "[CLS]";
    private final String SEP = "[SEP]";
    private final String PAD = "[PAD]";

    private EditText edt_input;
    private HashMap<String, Integer> tokenToIdMap;
    private HashMap<Integer, String> idToTokenMap;
    private HashMap<String, Integer> labelToIdMap;
    private HashMap<Integer, String> idToLabelMap;
    private Module module;
    private InputMethodManager inputMethodManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        edt_input = findViewById(R.id.edt_input);

        inputMethodManager = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);

        initLabelMapping();

        initVocabularyMapping();

        loadModel();

        initRunButton();
    }

    private void initLabelMapping() {
        labelToIdMap = new HashMap<String, Integer>();
        idToLabelMap = new HashMap<Integer, String>();

        try {
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(getAssets().open("label.txt"), StandardCharsets.UTF_8)
            );
            String mLine;

            while ((mLine = reader.readLine()) != null) {
                String[] split = mLine.split("<>");
                labelToIdMap.put(split[0], Integer.parseInt(split[1]));
                idToLabelMap.put(Integer.parseInt(split[1]), split[0]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void initRunButton() {
        binding.btnRun.setOnClickListener(view -> {
            inputMethodManager.hideSoftInputFromWindow(view.getWindowToken(), 0);

            try {
                ArrayList<Integer> tokenized = tokenize(edt_input.getText().toString());
                ArrayList<String> tokenized_string = decodeInputIds(tokenized);
                binding.txtTokenizedInput.setText(tokenized_string.toString());
                binding.txtInputIds.setText(tokenized.toString());
                ArrayList<Integer> classified_id = classifyTokens(tokenized);
                binding.txtClassId.setText(convertIdToLabel(classified_id).toString());
                binding.txtClassLabel.setText(groupLabel(tokenized_string, classified_id).toString());
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        });
    }

    private void loadModel() {
        try {
            module = Module.load(assetFilePath("model.pt"));
        } catch (IOException e) {
            Log.e("MainActiv", "Unable to load model", e);
        }
    }

    private void initVocabularyMapping() {
        tokenToIdMap = new HashMap<String, Integer>();
        idToTokenMap = new HashMap<Integer, String>();
        try {
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(getAssets().open("vocab.txt"), StandardCharsets.UTF_8)
            );
            String mLine;

            while ((mLine = reader.readLine()) != null) {
                String[] split = mLine.split("<>");
                tokenToIdMap.put(split[0], Integer.parseInt(split[1]));
                idToTokenMap.put(Integer.parseInt(split[1]), split[0]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    private ArrayList<String> decodeInputIds(ArrayList<Integer> input_ids) {
        ArrayList<String> result = new ArrayList<String>();
        for (Integer input_id : input_ids) {
            result.add(this.idToTokenMap.get(input_id));
        }
        return result;
    }

    private ArrayList<Integer> tokenize(String input) {
        // Clean input punctuation
        input = input.replaceAll("[^\\w\\s]", "");

        ArrayList<Integer> result = new ArrayList<Integer>();

        result.add(this.tokenToIdMap.get(this.CLS));

        String[] words = input.split("\\s+");

        for (String word : words) {
            result.addAll(tokenizeWordPiece(word));
        }

        result.add(this.tokenToIdMap.get(this.SEP));
        return result;
    }

    private String assetFilePath(String assetName) throws IOException {
        File file = new File(this.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = this.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    private ArrayList<Integer> tokenizeWordPiece(String input) {
        input = input.toLowerCase();
        ArrayList<Integer> result = new ArrayList<Integer>();

        if (this.tokenToIdMap.containsKey(input)) {
            result.add(this.tokenToIdMap.get(input));
            return result;
        }

        int input_length = input.length();
        for (int i = 0; i < input_length; i++) {
            if (this.tokenToIdMap.containsKey(input.substring(0, input_length - i - 1))) {
                result.add(this.tokenToIdMap.get(input.substring(0, input_length - i - 1)));
                String substring = input.substring(input_length - i - 1);

                int j = 0;
                while (j < substring.length()) {
                    String sub_substring = substring.substring(0, substring.length() - j);
                    if (this.tokenToIdMap.containsKey("##" + sub_substring)) {
                        result.add(this.tokenToIdMap.get("##" + sub_substring));
                        substring = substring.substring(substring.length() - j);
                        j = substring.length() - j;
                    } else if (j == substring.length() - 1) {
                        result.add(this.tokenToIdMap.get("##" + substring));
                        break;
                    } else j++;
                }
                break;
            }
        }

        return result;
    }

    private ArrayList<Integer> classifyTokens(ArrayList<Integer> token_ids) {
        LongBuffer tensor_buffer = Tensor.allocateLongBuffer(MODEL_INPUT_LENGTH);

        for (int i = 0; i < token_ids.size() && i < MODEL_INPUT_LENGTH; i++) {
            tensor_buffer.put(token_ids.get(i));
        }

        for (Integer token_id : token_ids) tensor_buffer.put(token_id);

        long[] arr = new long[]{1, MODEL_INPUT_LENGTH};

        Tensor tensor_input = Tensor.fromBlob(tensor_buffer, arr);
        IValue[] tensor_output = module.forward(IValue.from(tensor_input)).toTuple();

        Tensor tensor = tensor_output[0].toTensor();
        float[] logits = tensor.getDataAsFloatArray();

        ArrayList<Integer> result = new ArrayList<>();
        for (int i = 0; i < token_ids.size() && i < MODEL_INPUT_LENGTH; i++) {
            result.add(argmax(Arrays.copyOfRange(logits, i * NUM_LABEL, (i + 1) * NUM_LABEL)));
        }

        return result;
    }

    private ArrayList<String> convertIdToLabel(ArrayList<Integer> ids) {
        ArrayList<String> result = new ArrayList<>();
        for (Integer id : ids) {
            result.add(idToLabelMap.get(id));
        }

        return result;
    }

    private HashMap<String, ArrayList<String>> groupLabel(ArrayList<String> strings, ArrayList<Integer> ids) {
        HashMap<String, ArrayList<String>> result = new HashMap<>();

        for (int i = 0; i < ids.size(); i++) {
            //kalau id sekarang adalah start
            if (isBeginLabelId(ids.get(i))) {
                String current_label = Objects.requireNonNull(idToLabelMap.get(ids.get(i))).substring(2);
                ArrayList<String> current_strings = new ArrayList<>();
                current_strings.add(strings.get(i));

                while (true) {
                    if (++i >= ids.size()) break;

                    if (isFollowLabelId(ids.get(i))) {
                        current_strings.add(strings.get(i));
                        continue;
                    }

                    break;
                }

                String string = String.join(" ", current_strings).replaceAll(" ##", "")
                        .replaceAll("\\s+(?=\\p{Punct})", "");

                if (result.containsKey(current_label)) {
                    Objects.requireNonNull(result.get(current_label)).add(string);
                } else {
                    ArrayList<String> new_array = new ArrayList<>();
                    new_array.add(string);
                    result.put(current_label, new_array);
                }
            }
        }

        return result;
    }

    private boolean isBeginLabelId(int id) {
        return id <= BEGIN_LABEL_COUNT;
    }

    private boolean isFollowLabelId(int id) {
        return BEGIN_LABEL_COUNT < id && id < NUM_LABEL - 1;
    }

    private int argmax(float[] array) {
        int maxIdx = 0;
        float maxVal = -Float.MAX_VALUE;
        int id = 0;
        for (float j : array) {
            if (j > maxVal) {
                maxVal = j;
                maxIdx = id;
            }
            id++;
        }
        return maxIdx;
    }
}