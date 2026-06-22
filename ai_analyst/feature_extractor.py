import re
import socket
import urllib.parse


class URLFeatureExtractor:
    def __init__(self, url: str):
        self.url = url
        if not (url.startswith("http://") or url.startswith("https://")):
            # default to http if protocol omitted for parsing
            self.url = "http://" + url

        try:
            self.parsed_url = urllib.parse.urlparse(self.url)
            self.domain = self.parsed_url.netloc
        except Exception:
            self.parsed_url = None
            self.domain = url

    def get_features_dict(self) -> dict:
        """Extracts the 30 phishing features from the URL and returns a dictionary"""
        features = {}

        # 1. having_IP_Address
        # Check if domain matches IPv4 pattern
        ip_pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
        if re.match(ip_pattern, self.domain):
            features["having_IP_Address"] = -1
        else:
            features["having_IP_Address"] = 1

        # 2. URL_Length
        url_len = len(self.url)
        if url_len < 54:
            features["URL_Length"] = 1
        elif 54 <= url_len <= 75:
            features["URL_Length"] = 0
        else:
            features["URL_Length"] = -1

        # 3. Shortining_Service
        shortening_services = [
            "bit.ly",
            "goo.gl",
            "shorte.st",
            "go2l.ink",
            "x.co",
            "ow.ly",
            "t.co",
            "tinyurl",
            "tr.im",
            "is.gd",
            "cli.gs",
        ]
        if any(service in self.domain for service in shortening_services):
            features["Shortining_Service"] = -1
        else:
            features["Shortining_Service"] = 1

        # 4. having_At_Symbol
        if "@" in self.url:
            features["having_At_Symbol"] = -1
        else:
            features["having_At_Symbol"] = 1

        # 5. double_slash_redirecting
        # Standard HTTP/HTTPS is at position 6/7. If "//" occurs later, it's a redirect trick.
        if "//" in self.url[7:]:
            features["double_slash_redirecting"] = -1
        else:
            features["double_slash_redirecting"] = 1

        # 6. Prefix_Suffix
        # Phishing sites often add dashes to domain names (e.g. secure-paypal.com)
        if "-" in self.domain:
            features["Prefix_Suffix"] = -1
        else:
            features["Prefix_Suffix"] = 1

        # 7. having_Sub_Domain
        # Count dots in hostname. Less dots = safer.
        dots = self.domain.count(".")
        # remove www prefix
        if self.domain.startswith("www."):
            dots -= 1
        if dots <= 1:
            features["having_Sub_Domain"] = 1
        elif dots == 2:
            features["having_Sub_Domain"] = 0
        else:
            features["having_Sub_Domain"] = -1

        # 8. SSLfinal_State
        # Check if using HTTPS
        if self.parsed_url and self.parsed_url.scheme == "https":
            features["SSLfinal_State"] = 1
        else:
            features["SSLfinal_State"] = -1

        # 9. Domain_registeration_length
        # Heuristic: set to -1 if Prefix_Suffix is -1 or having_Sub_Domain is -1, else 1.
        if features["Prefix_Suffix"] == -1 or features["having_Sub_Domain"] == -1:
            features["Domain_registeration_length"] = -1
        else:
            features["Domain_registeration_length"] = 1

        # 10. Favicon
        # Heuristic: set to 1.
        features["Favicon"] = 1

        # 11. port
        # Check non-standard ports
        port = self.parsed_url.port if self.parsed_url else None
        if port and port not in [80, 443]:
            features["port"] = -1
        else:
            features["port"] = 1

        # 12. HTTPS_token
        # Check if domain contains "https" token (e.g., login-https-secure.com)
        if "https" in self.domain:
            features["HTTPS_token"] = -1
        else:
            features["HTTPS_token"] = 1

        # 13. Request_URL
        # Heuristic: set to 1
        features["Request_URL"] = 1

        # 14. URL_of_Anchor
        # Heuristic: if URL length is long, anchor urls are more likely mismatch.
        if url_len > 75:
            features["URL_of_Anchor"] = -1
        elif 54 <= url_len <= 75:
            features["URL_of_Anchor"] = 0
        else:
            features["URL_of_Anchor"] = 1

        # 15. Links_in_tags
        # Heuristic: default 1
        features["Links_in_tags"] = 1

        # 16. SFH (Server Form Handler)
        # Heuristic: default 1
        features["SFH"] = 1

        # 17. Submitting_to_email
        # Check if form submission contains mailto: or mail()
        if "mailto:" in self.url:
            features["Submitting_to_email"] = -1
        else:
            features["Submitting_to_email"] = 1

        # 18. Abnormal_URL
        # Check if domain is abnormal
        if len(self.domain) == 0:
            features["Abnormal_URL"] = -1
        else:
            features["Abnormal_URL"] = 1

        # 19. Redirect
        # Check redirect count (mocked to 1 redirection if long URL, else 0)
        if url_len > 100:
            features["Redirect"] = 0  # suspicious
        else:
            features["Redirect"] = 1  # legitimate

        # 20. on_mouseover
        features["on_mouseover"] = 1

        # 21. RightClick
        features["RightClick"] = 1

        # 22. popUpWidnow
        features["popUpWidnow"] = 1

        # 23. Iframe
        features["Iframe"] = 1

        # 24. age_of_domain
        # Heuristic: default 1
        features["age_of_domain"] = 1

        # 25. DNSRecord
        # Perform real DNS lookup test
        try:
            socket.gethostbyname(self.domain)
            features["DNSRecord"] = 1
        except Exception:
            features["DNSRecord"] = -1

        # 26. web_traffic
        # Heuristic: default 1
        features["web_traffic"] = 1

        # 27. Page_Rank
        features["Page_Rank"] = 1

        # 28. Google_Index
        features["Google_Index"] = 1

        # 29. Links_pointing_to_page
        features["Links_pointing_to_page"] = 1

        # 30. Statistical_report
        # If domain has known phishing patterns, set to -1
        suspicious_keywords = ["login", "verify", "secure", "update", "bank", "paypal", "signin", "support", "account"]
        if any(keyword in self.domain.lower() for keyword in suspicious_keywords) and features["Prefix_Suffix"] == -1:
            features["Statistical_report"] = -1
        else:
            features["Statistical_report"] = 1

        return features

    def get_features_list(self) -> list:
        features_dict = self.get_features_dict()
        # Order should match exactly schema columns (except target 'Result')
        from phishsentinel.constant.training_pipeline import SCHEMA_COLUMNS

        ordered_features = []
        for col in SCHEMA_COLUMNS:
            if col != "Result":
                ordered_features.append(features_dict.get(col, 1))
        return ordered_features
