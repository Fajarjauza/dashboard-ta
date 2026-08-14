"""
Nama tema sub-topik LDA — disalin PERSIS dari notebook 06a_topic_modeling_subaspek.ipynb,
cell [E1], hasil interpretasi manual penulis terhadap kata representatif tiap sub-topik.
"""

NAMA_SUBTOPIK = {
    "Individual|0": "Manfaat dan Kemudahan Aplikasi bagi Ibu Hamil",
    "Individual|1": "Pemantauan Tumbuh Kembang Janin dan Bayi",
    "Individual|2": "Kendala Penggunaan Awal Aplikasi",

    "Technical|0": "Ulasan Umum Campuran (Apresiasi dan Kendala Koneksi/Unduh)",
    "Technical|1": "Kesulitan Instalasi dan Pendaftaran, Rating Rendah",
    "Technical|2": "Error dan Aplikasi Lambat Setelah Update",
    "Technical|3": "Kendala Bahasa dan Login/Akses Aplikasi",
    "Technical|4": "Apresiasi Fitur Informasi dan Edukasi",
    "Technical|5": "Apresiasi Kelengkapan Fitur dan Keluhan Konten Komunitas",
    "Technical|6": "Kendala Membuka Aplikasi dan Login Gagal",
    "Technical|7": "Ulasan Umum Campuran (Apresiasi Singkat dan Keluhan Teknis Beragam)",
    "Technical|8": "Kesulitan Membuka/Mengunduh Aplikasi",
    "Technical|9": "Kekecewaan Fitur Berbayar dan Aplikasi Keluar Sendiri",
    "Technical|10": "Masalah Loading dan Login",
    "Technical|11": "Apresiasi Informasi dan Pengalaman Berbagi",
    "Technical|12": "Aplikasi Keluar Sendiri dan Gagal Dibuka (Parah)",
    "Technical|13": "Kebingungan Input Tanggal dan Usia Kehamilan",
    "Technical|14": "Keluhan Upgrade Memaksa Login Ulang dan Macet",
    "Technical|15": "Keluhan Iklan dan Kegagalan Sistem",

    "Social|0": "Berbagi Pengalaman dengan Sesama Ibu Hamil",
    "Social|1": "Manfaat Komunitas dan Berbagi Informasi",

    "Financial|0": "Keluhan Biaya Konsultasi dan Fitur Premium vs Gratis",
    "Financial|1": "Pertanyaan dan Kendala Seputar Tanggal Haid/HPHT",
    "Financial|2": "Kendala Login Setelah Update",
    "Financial|3": "Kegagalan Login dan Pembuatan Akun",
    "Financial|4": "Cara Klaim Hadiah/Pembayaran",
    "Financial|5": "Kesulitan Pembelian Fitur Berbayar",
    "Financial|6": "Kebingungan Setelah Update dan Transaksi Gagal",
    "Financial|7": "Kekecewaan Akun dan Fitur Lambat",
    "Financial|8": "Kendala Login dan Verifikasi Akun Berulang",
    "Financial|9": "Pertanyaan Seputar Member VIP dan Fitur Gratis",
    "Financial|10": "Aplikasi Lambat dan Gagal Dibuka Setelah Update",
    "Financial|11": "Kendala Pendaftaran Akun",
}


def nama_subtopik(aspek, topic_id):
    return NAMA_SUBTOPIK.get(f"{aspek}|{topic_id}", f"Sub-topik #{topic_id} (belum dinamai)")
