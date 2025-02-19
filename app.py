import spotipy
from spotipy.oauth2 import SpotifyOAuth
import cv2
from fer import FER

class EmotionDetector:
    def __init__(self):
        self.detector = FER()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_emotion(self):
        cap = cv2.VideoCapture(0)
        detected_emotion = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                emotions = self.detector.detect_emotions(face_roi)

                if emotions:
                    dominant_emotion = emotions[0]['emotions']
                    max_emotion = max(dominant_emotion, key=dominant_emotion.get)
                    cv2.putText(frame, f'Emotion: {max_emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    detected_emotion = max_emotion

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('Emotion Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('s') and detected_emotion:
                cap.release()
                cv2.destroyAllWindows()
                return detected_emotion
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None

        cap.release()
        cv2.destroyAllWindows()
        return None


class MusicRecommender:
    def __init__(self, client_id, client_secret, redirect_uri):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                            client_secret=client_secret,
                                                            redirect_uri=redirect_uri,
                                                            scope='user-read-playback-state user-modify-playback-state playlist-modify-private'))

    def get_music_recommendations(self, emotion):
        genre_map = {
            'happy': 'pop',
            'sad': 'sad',
            'angry': 'rock',
            'surprised': 'dance',
            'neutral': 'chill'
        }
        
        genre = genre_map.get(emotion.lower(), 'pop')
        results = self.sp.search(q=f'genre:"{genre}"', type='track', limit=5)

        tracks = []
        for track in results['tracks']['items']:
            tracks.append(track['id'])  # Collecting track IDs for the playlist

        return tracks

    def create_playlist(self, user_id, playlist_name, track_ids):
        playlist = self.sp.user_playlist_create(user_id, playlist_name, public=False)
        self.sp.user_playlist_add_tracks(user_id, playlist['id'], track_ids)
        return playlist['external_urls']['spotify']  # Return the playlist URL


if __name__ == "__main__":
    SPOTIPY_CLIENT_ID = ''
    SPOTIPY_CLIENT_SECRET = ''
    SPOTIPY_REDIRECT_URI = ''

    emotion_detector = EmotionDetector()
    music_recommender = MusicRecommender(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI)

    # Start emotion detection
    detected_emotion = emotion_detector.detect_emotion()

    if detected_emotion:
        print(f"Detected Emotion: {detected_emotion}")
        
        # Get recommendations based on the detected emotion
        recommended_tracks = music_recommender.get_music_recommendations(detected_emotion)

        # Create a playlist
        user_id = music_recommender.sp.current_user()['id']  # Get current user ID
        playlist_name = f"{detected_emotion.capitalize()} Playlist"
        playlist_url = music_recommender.create_playlist(user_id, playlist_name, recommended_tracks)

        print(f"Playlist created! Check it out here: {playlist_url}")
    else:
        print("No emotion detected, or user quit.")
