from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from requests import Response

from canvas_sdk.clients.aws.libraries.aws_s3 import AwsS3
from canvas_sdk.clients.aws.structures.aws_credentials import AwsCredentials
from canvas_sdk.clients.aws.structures.aws_s3_item import AwsS3Item
from canvas_sdk.tests.conftest import has_constants


def test_constants() -> None:
    """Test AwsS3 class constants values."""
    assert has_constants(
        AwsS3,
        {
            "ALGORITHM": "AWS4-HMAC-SHA256",
            "SAFE_CHARACTERS": "-._~",
            "SERVICE_NAME": "s3",
            "REQUEST_TYPE": "aws4_request",
            "UNSIGNED_PAYLOAD": "UNSIGNED-PAYLOAD",
        },
    )


def test__querystring() -> None:
    """Test _querystring method generates proper URL-encoded query strings."""
    tests = [
        # No params
        (None, ""),
        # Empty dict
        ({}, ""),
        # Single param
        ({"key1": "value1"}, "key1=value1"),
        # Multiple params (sorted)
        ({"key2": "value2", "key1": "value1"}, "key1=value1&key2=value2"),
        # Special characters - changed
        ({"key": "value with spaces"}, "key=value%20with%20spaces"),
        # Special characters - unchanged
        ({"key": "value-with.characters_acceptable~"}, "key=value-with.characters_acceptable~"),
    ]
    tested = AwsS3
    for params, expected in tests:
        result = tested._querystring(params)
        assert result == expected


def test__hmac_bytes() -> None:
    """Test _hmac_bytes method returns HMAC-SHA256 digest as bytes."""
    key = b"test_key"
    data = "test_data"
    tested = AwsS3
    result = tested._hmac_bytes(key, data)
    expected = b"F\xa5\xb2{~fr'\x1c\x99\x8fMy\xedF\x0f\xf0<\x88\xca\xcd15_\xfc\x16\x159\xe1ex$"
    assert result == expected
    assert len(result) == 32  # SHA256 produces 32 bytes


def test__hmac_str() -> None:
    """Test _hmac_str method returns HMAC-SHA256 digest as hex string."""
    key = b"test_key"
    data = "test_data"
    tested = AwsS3
    result = tested._hmac_str(key, data)
    expected = "46a5b27b7e6672271c998f4d79ed460ff03c88cacd31355ffc161539e1657824"
    assert result == expected
    assert len(result) == 64  # SHA256 hex produces 64 characters


@patch("canvas_sdk.clients.aws.libraries.aws_s3.datetime")
def test__amz_date_time(mock_datetime: MagicMock) -> None:
    """Test _amz_date_time method returns current UTC time in AWS format."""

    def reset_mocks() -> None:
        mock_datetime.reset_mock()

    mock_datetime.now.side_effect = [datetime(2025, 12, 1, 15, 7, 53, 123456, tzinfo=UTC)]
    tested = AwsS3
    result = tested._amz_date_time()
    expected = "20251201T150753Z"
    assert result == expected
    calls = [call.now(UTC)]
    assert mock_datetime.mock_calls == calls
    reset_mocks()


def test__amz_date_from() -> None:
    """Test _amz_date_from method extracts date part from AWS datetime string."""
    tests = [
        ("20251201T150753Z", "20251201"),
        ("20250112T150753Z", "20250112"),
    ]
    tested = AwsS3
    for idx, (amz_date_time, expected) in enumerate(tests):
        result = tested._amz_date_from(amz_date_time)
        assert result == expected, f"--> {idx}"


def test___init__() -> None:
    """Test __init__ method stores credentials."""
    credentials = AwsCredentials(
        key="theKey",
        secret="theSecret",
        region="theRegion",
        bucket="theBucket",
    )
    tested = AwsS3(credentials)
    assert tested.credentials == credentials


def test_is_ready() -> None:
    """Test is_ready method validates all required credentials are present."""
    tests = [
        # All credentials provided
        (
            AwsCredentials(
                key="theKey", secret="theSecret", region="theRegion", bucket="theBucket"
            ),
            True,
        ),
        # Missing key
        (AwsCredentials(key="", secret="theSecret", region="theRegion", bucket="theBucket"), False),
        # Missing secret
        (AwsCredentials(key="theKey", secret="", region="theRegion", bucket="theBucket"), False),
        # Missing region
        (AwsCredentials(key="theKey", secret="theSecret", region="", bucket="theBucket"), False),
        # Missing bucket
        (AwsCredentials(key="theKey", secret="theSecret", region="theRegion", bucket=""), False),
    ]
    for idx, (credentials, expected) in enumerate(tests):
        tested = AwsS3(credentials)
        result = tested.is_ready()
        assert result is expected, f"--> {idx}"


def test__get_host() -> None:
    """Test _get_host method generates S3 endpoint hostname."""
    credentials = AwsCredentials(
        key="theKey",
        secret="theSecret",
        region="theRegion",
        bucket="theBucket",
    )

    tested = AwsS3(credentials)
    result = tested._get_host()
    expected = "theBucket.s3.theRegion.amazonaws.com"
    assert result == expected


def test__get_signature_key() -> None:
    """Test _get_signature_key method generates AWS Signature V4 signing key and signature."""
    credentials = AwsCredentials(
        key="theKey",
        secret="theSecret",
        region="theRegion",
        bucket="theBucket",
    )
    tested = AwsS3(credentials)

    credential_scope, signature = tested._get_signature_key(
        "20251201T121520Z", "theCanonicalRequest"
    )

    expected = "20251201/theRegion/s3/aws4_request"
    assert credential_scope == expected
    expected = "2628651da3fda37e48588e87e8e01022aac97686de2cfc463ee3b2beb038605b"
    assert signature == expected
    assert len(signature) == 64  # SHA256 hex


@patch.object(AwsS3, "_headers_full")
def test__headers_with_params(mock_headers_full: MagicMock) -> None:
    """Test _headers_with_params method generates headers with query parameters."""

    def reset_mocks() -> None:
        mock_headers_full.reset_mock()

    credentials = AwsCredentials(
        key="theKey",
        secret="theSecret",
        region="theRegion",
        bucket="theBucket",
    )
    tested = AwsS3(credentials)

    mock_headers_full.side_effect = [{"header": "value"}]
    result = tested._headers_with_params("file.txt", {"key": "value"})
    expected = {"header": "value"}
    assert result == expected

    calls = [call("file.txt", None, {"key": "value"})]
    assert mock_headers_full.mock_calls == calls
    reset_mocks()


@patch.object(AwsS3, "_headers_full")
def test__headers_with_data(mock_headers_full: MagicMock) -> None:
    """Test _headers_with_data method generates headers with data payload."""

    def reset_mocks() -> None:
        mock_headers_full.reset_mock()

    credentials = AwsCredentials(
        key="theKey",
        secret="theSecret",
        region="theRegion",
        bucket="theBucket",
    )
    tested = AwsS3(credentials)

    mock_headers_full.side_effect = [{"header": "value"}]
    result = tested._headers_with_data("file.txt", (b"data", "text/plain"))
    expected = {"header": "value"}
    assert result == expected

    calls = [call("file.txt", (b"data", "text/plain"), None)]
    assert mock_headers_full.mock_calls == calls
    reset_mocks()


@patch.object(AwsS3, "_headers_full")
def test__headers(mock_headers_full: MagicMock) -> None:
    """Test _headers method generates headers for simple GET request."""

    def reset_mocks() -> None:
        mock_headers_full.reset_mock()

    credentials = AwsCredentials(
        key="theKey",
        secret="theSecret",
        region="theRegion",
        bucket="theBucket",
    )
    tested = AwsS3(credentials)

    mock_headers_full.side_effect = [{"header": "value"}]
    result = tested._headers("file.txt")
    expected = {"header": "value"}
    assert result == expected

    calls = [call("file.txt", None, None)]
    assert mock_headers_full.mock_calls == calls
    reset_mocks()


@patch.object(AwsS3, "_amz_date_time")
def test__headers_full(mock_amz_date_time: MagicMock) -> None:
    """Test _headers_full method generates complete AWS Signature V4 headers."""

    def reset_mocks() -> None:
        mock_amz_date_time.reset_mock()

    tests = [
        # GET request (no data)
        (
            "file.txt",
            None,
            None,
            "host",
            "d3308e8a1e18d272898efebcf9ef0e575738d9845c748ffae69818c830d01809",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        # PUT request with data
        (
            "file.txt",
            (b"content", "text/plain"),
            None,
            "content-type;host",
            "4326709b7dfd5f6a860f2e75a7ce4061f4c550da9ea9f18b526d78c063591c03",
            "ed7002b439e9ac845f22357d822bac1444730fbdb6016d3ec9432297b9ec9f73",
        ),
        # GET request with params
        (
            "file.txt",
            None,
            {"key": "value"},
            "host",
            "f0a1ba5845600643d8c73c64d3f4d90a403279b14a7e9b90d452a3efd44a6112",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
    ]

    credentials = AwsCredentials(
        key="theKey",
        secret="theSecret",
        region="theRegion",
        bucket="theBucket",
    )
    tested = AwsS3(credentials)
    for idx, (object_key, data, params, exp_signed, exp_signature, exp_sha256) in enumerate(tests):
        mock_amz_date_time.side_effect = ["20251201T121543Z"]
        result = tested._headers_full(object_key, data, params)

        authorization = (
            "AWS4-HMAC-SHA256 Credential=theKey/20251201/theRegion/s3/aws4_request, "
            f"SignedHeaders={exp_signed};x-amz-content-sha256;x-amz-date, "
            f"Signature={exp_signature}"
        )

        expected = {
            "Authorization": authorization,
            "Host": "theBucket.s3.theRegion.amazonaws.com",
            "x-amz-content-sha256": exp_sha256,
            "x-amz-date": "20251201T121543Z",
        }
        assert result == expected, f"--> {idx}"
        calls = [call()]
        assert mock_amz_date_time.mock_calls == calls
        reset_mocks()


@patch("canvas_sdk.clients.aws.libraries.aws_s3.Http")
@patch.object(AwsS3, "_headers")
@patch.object(AwsS3, "is_ready")
def test_access_s3_object(
    mock_is_ready: MagicMock,
    mock_headers: MagicMock,
    mock_http: MagicMock,
) -> None:
    """Test access_s3_object method downloads objects from S3."""

    def reset_mocks() -> None:
        mock_is_ready.reset_mock()
        mock_headers.reset_mock()
        mock_http.reset_mock()

    credentials = AwsCredentials(
        key="theKey",
        secret="theSecret",
        region="theRegion",
        bucket="theBucket",
    )

    tested = AwsS3(credentials)

    # successful access
    mock_is_ready.side_effect = [True]
    mock_headers.side_effect = [{"Host": "theHost", "Authorization": "..."}]
    response = Response()
    mock_http.return_value.get.side_effect = [response]

    result = tested.access_s3_object("path/to/file.txt")
    assert result is response

    calls = [call()]
    assert mock_is_ready.mock_calls == calls
    calls = [call("path/to/file.txt")]
    assert mock_headers.mock_calls == calls
    calls = [
        call("https://theHost/"),
        call().get(url="path/to/file.txt", headers={"Host": "theHost", "Authorization": "..."}),
    ]
    assert mock_http.mock_calls == calls
    reset_mocks()

    # not ready
    mock_is_ready.side_effect = [False]
    mock_headers.side_effect = []
    mock_http.return_value.get.side_effect = []

    result = tested.access_s3_object("path/to/file.txt")
    assert result is None

    calls = [call()]
    assert mock_is_ready.mock_calls == calls
    assert mock_headers.mock_calls == []
    assert mock_http.mock_calls == []
    reset_mocks()


@patch.object(AwsS3, "upload_binary_to_s3")
def test_upload_text_to_s3(mock_upload_binary: MagicMock) -> None:
    """Test upload_text_to_s3 method uploads text data to S3."""

    def reset_mocks() -> None:
        mock_upload_binary.reset_mock()

    credentials = AwsCredentials(
        key="theKey",
        secret="theSecret",
        region="theRegion",
        bucket="theBucket",
    )

    tested = AwsS3(credentials)
    response = Response()
    mock_upload_binary.side_effect = [response]

    result = tested.upload_text_to_s3("file.txt", "Hello World")
    assert result is response

    calls = [call("file.txt", b"Hello World", "text/plain")]
    assert mock_upload_binary.mock_calls == calls
    reset_mocks()


@patch("canvas_sdk.clients.aws.libraries.aws_s3.Http")
@patch.object(AwsS3, "_headers_with_data")
@patch.object(AwsS3, "is_ready")
def test_upload_binary_to_s3(
    mock_is_ready: MagicMock,
    mock_headers_with_data: MagicMock,
    mock_http: MagicMock,
) -> None:
    """Test upload_binary_to_s3 method uploads binary data to S3."""

    def reset_mocks() -> None:
        mock_is_ready.reset_mock()
        mock_headers_with_data.reset_mock()
        mock_http.reset_mock()

    credentials = AwsCredentials(
        key="theKey",
        secret="theSecret",
        region="theRegion",
        bucket="theBucket",
    )

    tested = AwsS3(credentials)

    # successful upload
    mock_is_ready.side_effect = [True]
    mock_headers_with_data.side_effect = [{"Host": "theHost", "Authorization": "..."}]
    response = Response()
    mock_http.return_value.put.side_effect = [response]

    binary_data = b"binary content"
    result = tested.upload_binary_to_s3("file.bin", binary_data, "application/octet-stream")
    assert result is response

    calls = [call()]
    assert mock_is_ready.mock_calls == calls
    calls = [call("file.bin", (b"binary content", "application/octet-stream"))]
    assert mock_headers_with_data.mock_calls == calls
    calls = [
        call("https://theHost/"),
        call().put(
            url="file.bin",
            headers={
                "Host": "theHost",
                "Authorization": "...",
                "Content-Type": "application/octet-stream",
                "Content-Length": "14",
            },
            data=b"binary content",
        ),
    ]
    assert mock_http.mock_calls == calls
    reset_mocks()

    # not ready
    mock_is_ready.side_effect = [False]
    mock_headers_with_data.side_effect = []
    mock_http.return_value.put.side_effect = []

    result = tested.upload_binary_to_s3("file.bin", binary_data, "application/octet-stream")
    assert result is None

    calls = [call()]
    assert mock_is_ready.mock_calls == calls
    assert mock_headers_with_data.mock_calls == []
    assert mock_http.mock_calls == []
    reset_mocks()


@patch("canvas_sdk.clients.aws.libraries.aws_s3.Http")
@patch.object(AwsS3, "_headers_with_params")
@patch.object(AwsS3, "_querystring")
@patch.object(AwsS3, "is_ready")
def test_list_s3_objects(
    mock_is_ready: MagicMock,
    mock_querystring: MagicMock,
    mock_headers_with_params: MagicMock,
    mock_http: MagicMock,
) -> None:
    """Test list_s3_objects method lists all S3 objects with pagination support."""

    def reset_mocks() -> None:
        mock_is_ready.reset_mock()
        mock_querystring.reset_mock()
        mock_headers_with_params.reset_mock()
        mock_http.reset_mock()

    credentials = AwsCredentials(
        key="theKey",
        secret="theSecret",
        region="theRegion",
        bucket="theBucket",
    )

    tested = AwsS3(credentials)

    # is ready
    responses = [
        SimpleNamespace(
            status_code=200,
            content=b"""<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult>
        <IsTruncated>true</IsTruncated>
        <NextContinuationToken>token123</NextContinuationToken>
        <Contents>
            <Key>file1.txt</Key>
            <Size>1024</Size>
            <LastModified>2025-12-01T12:16:12.123Z</LastModified>
        </Contents>
        <Contents>
            <Key>file2.txt</Key>
            <Size>2048</Size>
            <LastModified>2025-12-02T13:07:41.456Z</LastModified>
        </Contents>
    </ListBucketResult>""",
        ),
        SimpleNamespace(
            status_code=200,
            content=b"""<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult>
        <IsTruncated>false</IsTruncated>
        <Contents>
            <Key>file3.txt</Key>
            <Size>1750</Size>
            <LastModified>2025-12-02T13:23:11.789Z</LastModified>
        </Contents>
        <Contents>
            <Key>file4.txt</Key>
            <Comment>invalid key</Comment>
        </Contents>
    </ListBucketResult>""",
        ),
    ]

    mock_is_ready.side_effect = [True]
    mock_headers_with_params.side_effect = [{"Host": "theHost1"}, {"Host": "theHost2"}]
    mock_querystring.side_effect = ["theQueryString1", "theQueryString2"]
    mock_http.return_value.get.side_effect = responses

    result = tested.list_s3_objects("test")

    expected = [
        AwsS3Item(
            key="file1.txt",
            size=1024,
            last_modified=datetime(2025, 12, 1, 12, 16, 12, 123000, tzinfo=UTC),
        ),
        AwsS3Item(
            key="file2.txt",
            size=2048,
            last_modified=datetime(2025, 12, 2, 13, 7, 41, 456000, tzinfo=UTC),
        ),
        AwsS3Item(
            key="file3.txt",
            size=1750,
            last_modified=datetime(2025, 12, 2, 13, 23, 11, 789000, tzinfo=UTC),
        ),
    ]
    assert result == expected

    calls = [call()]
    assert mock_is_ready.mock_calls == calls
    calls = [
        call("", {"list-type": 2, "prefix": "test"}),
        call("", {"list-type": 2, "prefix": "test", "continuation-token": "token123"}),
    ]
    assert mock_headers_with_params.mock_calls == calls
    calls = [
        call({"list-type": 2, "prefix": "test"}),
        call({"list-type": 2, "prefix": "test", "continuation-token": "token123"}),
    ]
    assert mock_querystring.mock_calls == calls
    calls = [
        call("https://theHost1?theQueryString1"),
        call().get(url="", headers={"Host": "theHost1"}),
        call("https://theHost2?theQueryString2"),
        call().get(url="", headers={"Host": "theHost2"}),
    ]
    assert mock_http.mock_calls == calls
    reset_mocks()
    # is NOT ready
    mock_is_ready.side_effect = [False]
    mock_headers_with_params.side_effect = []
    mock_querystring.side_effect = []
    mock_http.return_value.get.side_effect = []

    result = tested.list_s3_objects("test")
    assert result is None

    calls = [call()]
    assert mock_is_ready.mock_calls == calls
    assert mock_headers_with_params.mock_calls == []
    assert mock_querystring.mock_calls == []
    assert mock_http.mock_calls == []
    reset_mocks()

    # error response
    responses = [SimpleNamespace(status_code=404, text="Access Denied")]
    mock_is_ready.side_effect = [True]
    mock_headers_with_params.side_effect = [{"Host": "theHost1"}]
    mock_querystring.side_effect = ["theQueryString1"]
    mock_http.return_value.get.side_effect = responses
    with pytest.raises(Exception, match="S3 response status code 404 with body Access Denied"):
        tested.list_s3_objects("test")

    calls = [call()]
    assert mock_is_ready.mock_calls == calls
    calls = [call("", {"list-type": 2, "prefix": "test"})]
    assert mock_headers_with_params.mock_calls == calls
    calls = [call({"list-type": 2, "prefix": "test"})]
    assert mock_querystring.mock_calls == calls
    calls = [
        call("https://theHost1?theQueryString1"),
        call().get(url="", headers={"Host": "theHost1"}),
    ]
    assert mock_http.mock_calls == calls
    reset_mocks()


@patch.object(AwsS3, "_get_signature_key")
@patch.object(AwsS3, "_amz_date_time")
@patch.object(AwsS3, "_querystring")
@patch.object(AwsS3, "is_ready")
def test_generate_presigned_url(
    mock_is_ready: MagicMock,
    mock_querystring: MagicMock,
    mock_amz_date_time: MagicMock,
    mock_get_signature_key: MagicMock,
) -> None:
    """Test generate_presigned_url method creates temporary access URLs for S3 objects."""

    def reset_mocks() -> None:
        mock_is_ready.reset_mock()
        mock_querystring.reset_mock()
        mock_amz_date_time.reset_mock()
        mock_get_signature_key.reset_mock()

    credentials = AwsCredentials(
        key="theKey",
        secret="theSecret",
        region="theRegion",
        bucket="theBucket",
    )

    tested = AwsS3(credentials)

    # successful generation
    mock_is_ready.side_effect = [True]
    mock_querystring.side_effect = ["theQueryString1", "theQueryString2"]
    mock_amz_date_time.side_effect = ["20251201T123456Z"]
    mock_get_signature_key.side_effect = [("theCredentialScope", "theSignature")]

    result = tested.generate_presigned_url("file.txt", 4321)
    expected = "https://theBucket.s3.theRegion.amazonaws.com/file.txt?theQueryString2"
    assert result == expected

    calls = [call()]
    assert mock_is_ready.mock_calls == calls
    calls = [
        call(
            {
                "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
                "X-Amz-Credential": "theKey/20251201/theRegion/s3/aws4_request",
                "X-Amz-Date": "20251201T123456Z",
                "X-Amz-Expires": "4321",
                "X-Amz-SignedHeaders": "host",
                "X-Amz-Content-Sha256": "UNSIGNED-PAYLOAD",
                "X-Amz-Signature": "theSignature",
            }
        ),
        call(
            {
                "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
                "X-Amz-Credential": "theKey/20251201/theRegion/s3/aws4_request",
                "X-Amz-Date": "20251201T123456Z",
                "X-Amz-Expires": "4321",
                "X-Amz-SignedHeaders": "host",
                "X-Amz-Content-Sha256": "UNSIGNED-PAYLOAD",
                "X-Amz-Signature": "theSignature",
            }
        ),
    ]
    assert mock_querystring.mock_calls == calls
    calls = [call()]
    assert mock_amz_date_time.mock_calls == calls
    calls = [
        call(
            "20251201T123456Z",
            "GET\n/file.txt\ntheQueryString1\nhost:theBucket.s3.theRegion.amazonaws.com\n\nhost\nUNSIGNED-PAYLOAD",
        )
    ]
    assert mock_get_signature_key.mock_calls == calls
    reset_mocks()

    # not ready
    mock_is_ready.side_effect = [False]
    mock_querystring.side_effect = []
    mock_amz_date_time.side_effect = []
    mock_get_signature_key.side_effect = []

    result = tested.generate_presigned_url("file.txt", 4321)
    assert result is None

    calls = [call()]
    assert mock_is_ready.mock_calls == calls
    assert mock_querystring.mock_calls == []
    assert mock_amz_date_time.mock_calls == []
    assert mock_get_signature_key.mock_calls == []
    reset_mocks()
