from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

video_id = "wjZofJX0v4M"  # 3Blue1Brown video

try:
    # Retrieve the available transcripts for this video
    transcript_list = YouTubeTranscriptApi.list(video_id)

    # Iterate over all available transcripts
    for transcript in transcript_list:
        # Print transcript metadata
        print(
            transcript.video_id,
            transcript.language,
            transcript.language_code,
            transcript.is_generated,          # manually created or auto-generated
            transcript.is_translatable,       # can it be translated
            transcript.translation_languages, # available translation languages
        )

        # Fetch the actual transcript data
        data = transcript.fetch()
        print("Transcript sample:", data[:3])  # first 3 entries

        # Translate transcript (if possible)
        if transcript.is_translatable:
            translated = transcript.translate("en").fetch()
            print("Translated sample:", translated[:3])

    # You can also directly filter transcripts
    transcript = transcript_list.find_transcript(["de", "en"])
    manual_transcript = transcript_list.find_manually_created_transcript(["de", "en"])
    auto_transcript = transcript_list.find_generated_transcript(["de", "en"])

    # Flatten English transcript into plain text (if found)
    en_transcript = transcript.fetch()
    text = " ".join([t["text"] for t in en_transcript])
    print("Full English transcript:", text)

except TranscriptsDisabled:
    print("No captions available for this video.")
