import { useCallback, useState } from 'react';
import PropTypes from 'prop-types';
import clsx from 'clsx';

const filePropType = typeof File === 'undefined' ? PropTypes.any : PropTypes.instanceOf(File);

const UploadDropzone = ({ acceptedExtensions, previewUrl, file, onFileSelect }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((event) => {
    event.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((event) => {
    event.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      setIsDragging(false);
      const [droppedFile] = event.dataTransfer.files ?? [];
      if (droppedFile) {
        onFileSelect(droppedFile);
      }
    },
    [onFileSelect]
  );

  const handleInputChange = useCallback(
    (event) => {
      const [nextFile] = event.target.files ?? [];
      if (nextFile) {
        onFileSelect(nextFile);
      }
    },
    [onFileSelect]
  );

  return (
    <div className="dropzone-wrapper">
      <div
        className={clsx('dropzone', isDragging && 'dropzone--active')}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            event.currentTarget.querySelector('input[type="file"]').click();
          }
        }}
        aria-label="Upload a face photo"
      >
        <input
          id="file-upload"
          className="dropzone__input"
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleInputChange}
        />

        <div className="dropzone__content">
          <span className="dropzone__icon" aria-hidden="true">
            📷
          </span>
          <p className="dropzone__title">Drop your photo here</p>
          <p className="dropzone__subtitle">or click to browse your files</p>
          <p className="dropzone__hint">Supports {acceptedExtensions}</p>
          {file && (
            <p className="dropzone__filename" aria-live="polite">
              Selected: {file.name}
            </p>
          )}
        </div>
      </div>

      {previewUrl && (
        <figure className="preview" aria-label="Selected image preview">
          <img src={previewUrl} alt="Selected face preview" />
        </figure>
      )}
    </div>
  );
};

UploadDropzone.propTypes = {
  acceptedExtensions: PropTypes.string.isRequired,
  previewUrl: PropTypes.string,
  file: filePropType,
  onFileSelect: PropTypes.func.isRequired
};

UploadDropzone.defaultProps = {
  previewUrl: '',
  file: null
};

export default UploadDropzone;
