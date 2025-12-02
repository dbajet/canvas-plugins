from canvas_sdk.clients.aws.structures.aws_credentials import AwsCredentials
from canvas_sdk.tests.conftest import is_namedtuple


def test_class() -> None:
    """Test AwsCredentials is a NamedTuple with correct fields and types."""
    assert is_namedtuple(
        AwsCredentials,
        {
            "key": str,
            "secret": str,
            "region": str,
            "bucket": str,
        },
    )
