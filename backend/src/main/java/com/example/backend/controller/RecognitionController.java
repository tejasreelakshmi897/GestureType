
package com.example.backend.controller;

import java.util.List;
import java.util.Map;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.backend.entity.Recognition;
import com.example.backend.repository.RecognitionRepository;

@RestController
@RequestMapping("/api/recognitions")

@CrossOrigin(origins = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://gesturetype.netlify.app"
})

public class RecognitionController {

    private final RecognitionRepository recognitionRepository;

    public RecognitionController(
            RecognitionRepository recognitionRepository) {
        this.recognitionRepository = recognitionRepository;
    }

    // Save recognized text
    @PostMapping
    public ResponseEntity<?> saveRecognition(
            @RequestBody Map<String, String> request) {

        String text = request.get("recognizedText");

        if (text == null || text.isBlank()) {
            return ResponseEntity.badRequest()
                    .body(Map.of(
                        "error", "recognizedText cannot be empty"
                    ));
        }

        Recognition recognition =
                new Recognition(text.trim());

        Recognition saved =
                recognitionRepository.save(recognition);

        return ResponseEntity.status(HttpStatus.CREATED)
                .body(saved);
    }

    // Get all recognition history
    @GetMapping
    public List<Recognition> getAllRecognitions() {
        return recognitionRepository.findAll();
    }

    // Get one recognition by ID
    @GetMapping("/{id}")
    public ResponseEntity<Recognition> getRecognitionById(
            @PathVariable Long id) {

        return recognitionRepository.findById(id)
                .map(ResponseEntity::ok)
                .orElseGet(() ->
                    ResponseEntity.notFound().build()
                );
    }

    // Delete one recognition by ID
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteRecognition(
            @PathVariable Long id) {

        if (!recognitionRepository.existsById(id)) {
            return ResponseEntity.notFound().build();
        }

        recognitionRepository.deleteById(id);
        return ResponseEntity.noContent().build();
    }

    // Delete all recognition history
    @DeleteMapping
    public ResponseEntity<Void> deleteAllRecognitions() {
        recognitionRepository.deleteAll();
        return ResponseEntity.noContent().build();
    }
}
