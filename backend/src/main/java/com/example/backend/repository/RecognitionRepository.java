
package com.example.backend.repository;

import com.example.backend.entity.Recognition;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface RecognitionRepository
        extends JpaRepository<Recognition, Long> {
}
