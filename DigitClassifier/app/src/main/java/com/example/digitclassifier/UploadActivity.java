package com.example.digitclassifier;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.Toast;

import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.FormBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
//  References
//  https://www.android.com/
//  https://developer.android.com/guide
//  https://developer.android.com/training/camera/photobasics
//  https://developer.android.com/develop/ui/views/components/spinner
//  https://www.tutlane.com/tutorial/android/android-spinner-dropdown-list-with-examples
public class UploadActivity extends AppCompatActivity {
//    private Spinner spinnerTag;
//    private String tag;
    private String BASE_URL = APIContract.BASE_URL;
    private Button upload;
    private String imageNameValue = MainActivity.getImageName();
    private String imageBase64 = MainActivity.getImageBase();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_upload_page);
        upload = findViewById(R.id.uploadBtn);
        upload.setOnClickListener(view -> sendRequest());
//        spinnerTag = findViewById(R.id.tagSelect);
////        String[] tagNames = getResources().getStringArray(R.array.imageTags);
//        ArrayAdapter adapter = new ArrayAdapter(this, android.R.layout.simple_spinner_item, tagNames);
//        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
//        spinnerTag.setAdapter(adapter);
//        spinnerTag.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
//            @Override
//            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
//                tag = parent.getItemAtPosition(pos).toString();
//            }
//
//            @Override
//            public void onNothingSelected(AdapterView<?> adapterView) {
//            }
//        });
    }

    private void sendRequest() {
        OkHttpClient client = new OkHttpClient();
//        RequestBody formbody = new FormBody.Builder().add("categoryName",categoryValue).add("imageName",imageNameValue).add("imageBase",imageBase64).build();
        RequestBody formbody = new FormBody.Builder().add("imageName",imageNameValue).add("imageBase",imageBase64).build();
        Request request = new Request.Builder().url(BASE_URL).post(formbody).build();
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull final IOException e) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(UploadActivity.this, e.getMessage(), Toast.LENGTH_LONG).show();
                    }
                });
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            Toast.makeText(UploadActivity.this, response.body().string(), Toast.LENGTH_LONG).show();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                });
                Intent intent = new Intent(UploadActivity.this, MainActivity.class);
                intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                startActivity(intent);
            }
        });

    }
}